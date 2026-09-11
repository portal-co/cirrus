/**
 * A deliberately small execution seam for circuit-fused Volar Boolar IR.
 *
 * Wires are opaque and alias-safe: a context may represent a wire as a
 * plaintext boolean, a garbled label, or another symbolic value.  The core
 * never inspects an unknown wire.  It only carries the `known` fact needed by
 * Cirrus-style storage adapters to simplify public address bits.
 */

export type VarId = number;
export type StorageId = number;
export type LaneId = number;

export interface StorageAddressBit<W> {
  readonly wire: W;
  readonly known: boolean | undefined;
}

export interface BoolarContext<W, Store> {
  create(value: boolean): W;
  and(left: W, right: W): W;
  or(left: W, right: W): W;
  xor(left: W, right: W): W;
  storageRead(store: Store, address: readonly StorageAddressBit<W>[]): W;
  storageWrite(store: Store, address: readonly StorageAddressBit<W>[], value: W): void;
}

/** Caller-owned storage namespace, mirroring Cirrus's `StorageBank`. */
export interface StorageBank<Store> {
  readonly storage: StorageId;
  readonly lane: LaneId;
  readonly addressBits: number;
  readonly value: Store;
}

export interface ExternalBitRegistry<W, Store> {
  oracleBit(
    context: BoolarContext<W, Store>,
    name: string,
    args: readonly W[],
    bit: number,
    occurrence: bigint,
  ): W;

  rngBit(
    context: BoolarContext<W, Store>,
    name: string,
    bit: number,
    occurrence: bigint,
  ): W;

  actionStoreBit(
    context: BoolarContext<W, Store>,
    call: ActionStoreBit,
    guard: W,
    args: readonly W[],
    fallback: W,
    address: readonly StorageAddressBit<W>[],
    bank: StorageBank<Store>,
  ): void;
}

export type BoolarStatement =
  | { readonly kind: "zero" }
  | { readonly kind: "one" }
  | { readonly kind: "and" | "or" | "xor"; readonly left: VarId; readonly right: VarId }
  | { readonly kind: "not"; readonly input: VarId }
  | { readonly kind: "storage_read"; readonly storage: StorageId; readonly lane: LaneId; readonly address: readonly VarId[] }
  | { readonly kind: "storage_write"; readonly storage: StorageId; readonly lane: LaneId; readonly source: VarId; readonly address: readonly VarId[] }
  | OracleBit
  | RngBit
  | ActionStoreBit
  // These variants are retained so the parser faithfully models old text
  // artifacts.  Cirrus's eager executor rejects them, and this executor does
  // the same until their aggregate external contract has an implementation.
  | { readonly kind: "oracle_call"; readonly name: string; readonly args: readonly VarId[]; readonly numBits: number }
  | { readonly kind: "oracle_projected_bit"; readonly call: VarId; readonly bit: number }
  | { readonly kind: "action_call"; readonly name: string; readonly guard: VarId; readonly args: readonly VarId[]; readonly fallback: readonly VarId[]; readonly numBits: number }
  | { readonly kind: "action_bit"; readonly call: VarId; readonly bit: number }
  | { readonly kind: "rng"; readonly name: string };

export interface OracleBit {
  readonly kind: "oracle_bit";
  readonly name: string;
  readonly args: readonly VarId[];
  readonly bit: number;
  readonly occurrence: bigint;
}

export interface RngBit {
  readonly kind: "rng_bit";
  readonly name: string;
  readonly bit: number;
  readonly occurrence: bigint;
}

export interface ActionStoreBit {
  readonly kind: "action_store_bit";
  readonly name: string;
  readonly guard: VarId;
  readonly args: readonly VarId[];
  readonly fallback: VarId;
  readonly storage: StorageId;
  readonly lane: LaneId;
  readonly address: readonly VarId[];
  readonly bit: number;
  readonly occurrence: bigint;
}

export interface BoolarTarget {
  readonly block: "return" | number;
  readonly args: readonly VarId[];
}

export type BoolarTerminator =
  | { readonly kind: "jmp"; readonly target: BoolarTarget }
  | { readonly kind: "cond_jmp"; readonly value: VarId; readonly thenTarget: BoolarTarget; readonly elseTarget: BoolarTarget };

export interface BoolarBlock {
  readonly params: number;
  readonly statements: readonly BoolarStatement[];
  readonly terminator: BoolarTerminator;
}

export interface PreInitSegment {
  readonly storage: StorageId;
  readonly lane: LaneId;
  /** LSB-first static base address. */
  readonly address: readonly boolean[];
  readonly data: readonly boolean[];
}

export interface BoolarBlocks {
  readonly blocks: readonly BoolarBlock[];
  readonly preInit: readonly PreInitSegment[];
}

export interface BoolarCircuit {
  readonly params: number;
  readonly statements: readonly BoolarStatement[];
  readonly preInit: readonly PreInitSegment[];
  readonly outputs: readonly VarId[];
}

export class BoolarError extends Error {
  constructor(
    readonly code:
      | "parse"
      | "input_arity"
      | "unsupported_control_flow"
      | "unsupported_statement"
      | "undefined_variable"
      | "storage"
      | "external",
    message: string,
    readonly detail: Readonly<Record<string, unknown>> = {},
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "BoolarError";
  }
}

/** Convert a single-block `jmp return` program into Cirrus's BCircuit shape. */
export function fuse(blocks: BoolarBlocks): BoolarCircuit {
  if (blocks.blocks.length !== 1) {
    throw new BoolarError("unsupported_control_flow", "Boolar execution requires exactly one block", { found: blocks.blocks.length });
  }
  const [block] = blocks.blocks;
  if (block.terminator.kind !== "jmp" || block.terminator.target.block !== "return") {
    throw new BoolarError("unsupported_control_flow", "Boolar execution requires a jmp return terminator");
  }
  const limit = block.params + block.statements.length;
  for (const output of block.terminator.target.args) {
    if (!Number.isInteger(output) || output < 0 || output >= limit) {
      throw new BoolarError("undefined_variable", "Boolar output refers to an undefined variable", { output, available: limit });
    }
  }
  return {
    params: block.params,
    statements: block.statements,
    preInit: blocks.preInit,
    outputs: block.terminator.target.args,
  };
}

interface Value<W> {
  readonly wire: W;
  readonly known: boolean | undefined;
}

export interface ExecuteOptions<W, Store> {
  readonly context: BoolarContext<W, Store>;
  readonly banks?: readonly StorageBank<Store>[];
  readonly externals?: ExternalBitRegistry<W, Store>;
}

/** Execute a fused Boolar circuit through a Cirrus-shaped context. */
export function execute<W, Store>(
  circuit: BoolarCircuit,
  inputs: readonly W[],
  options: ExecuteOptions<W, Store>,
): W[] {
  if (inputs.length !== circuit.params) {
    throw new BoolarError("input_arity", "incorrect Boolar input arity", { expected: circuit.params, found: inputs.length });
  }
  const banks = options.banks ?? [];
  const bankMap = validateBanks(banks);
  validateCircuitStorage(circuit, bankMap);
  initializeStorage(circuit.preInit, options.context, bankMap);

  const values: Value<W>[] = inputs.map((wire) => ({ wire, known: undefined }));
  let canonicalOne: W | undefined;

  const at = (id: VarId): Value<W> => {
    const value = values[id];
    if (value === undefined) {
      throw new BoolarError("undefined_variable", "Boolar statement refers to an undefined variable", { variable: id, available: values.length });
    }
    return value;
  };
  const address = (ids: readonly VarId[]): StorageAddressBit<W>[] => ids.map((id) => {
    const value = at(id);
    return { wire: value.wire, known: value.known };
  });
  const bankFor = (storage: StorageId, lane: LaneId): StorageBank<Store> => {
    const bank = bankMap.get(bankKey(storage, lane));
    if (bank === undefined) {
      throw new BoolarError("storage", "Boolar circuit requires a missing storage bank", { storage, lane });
    }
    return bank;
  };
  const one = (): W => {
    canonicalOne ??= options.context.create(true);
    return canonicalOne;
  };

  for (const statement of circuit.statements) {
    let value: Value<W>;
    switch (statement.kind) {
      case "zero":
        value = { wire: options.context.create(false), known: false };
        break;
      case "one":
        value = { wire: one(), known: true };
        break;
      case "and":
        value = applyAnd(options.context, at(statement.left), at(statement.right));
        break;
      case "or":
        value = applyOr(options.context, at(statement.left), at(statement.right));
        break;
      case "xor":
        value = applyXor(options.context, at(statement.left), at(statement.right));
        break;
      case "not": {
        const input = at(statement.input);
        value = input.known === undefined
          ? { wire: options.context.xor(input.wire, one()), known: undefined }
          : { wire: options.context.create(!input.known), known: !input.known };
        break;
      }
      case "storage_read": {
        const bank = bankFor(statement.storage, statement.lane);
        const bits = address(statement.address);
        assertAddressWidth(bank, bits.length);
        value = { wire: options.context.storageRead(bank.value, bits), known: undefined };
        break;
      }
      case "storage_write": {
        const bank = bankFor(statement.storage, statement.lane);
        const bits = address(statement.address);
        assertAddressWidth(bank, bits.length);
        options.context.storageWrite(bank.value, bits, at(statement.source).wire);
        value = { wire: options.context.create(false), known: false };
        break;
      }
      case "oracle_bit": {
        const registry = requiredExternal(options.externals, "oracle", statement.name);
        try {
          value = {
            wire: registry.oracleBit(options.context, statement.name, statement.args.map((id) => at(id).wire), statement.bit, statement.occurrence),
            known: undefined,
          };
        } catch (cause) {
          throw new BoolarError("external", `oracle '${statement.name}' failed`, { bit: statement.bit, occurrence: statement.occurrence }, cause);
        }
        break;
      }
      case "rng_bit": {
        const registry = requiredExternal(options.externals, "rng", statement.name);
        try {
          value = { wire: registry.rngBit(options.context, statement.name, statement.bit, statement.occurrence), known: undefined };
        } catch (cause) {
          throw new BoolarError("external", `RNG '${statement.name}' failed`, { bit: statement.bit, occurrence: statement.occurrence }, cause);
        }
        break;
      }
      case "action_store_bit": {
        const registry = requiredExternal(options.externals, "action", statement.name);
        const bank = bankFor(statement.storage, statement.lane);
        const bits = address(statement.address);
        assertAddressWidth(bank, bits.length);
        try {
          registry.actionStoreBit(
            options.context,
            statement,
            at(statement.guard).wire,
            statement.args.map((id) => at(id).wire),
            at(statement.fallback).wire,
            bits,
            bank,
          );
        } catch (cause) {
          throw new BoolarError("external", `action '${statement.name}' failed`, { bit: statement.bit, occurrence: statement.occurrence }, cause);
        }
        value = { wire: options.context.create(false), known: false };
        break;
      }
      default:
        throw new BoolarError("unsupported_statement", `Boolar statement '${statement.kind}' is not supported by the fused executor`);
    }
    values.push(value);
  }
  return circuit.outputs.map((output) => at(output).wire);
}

function requiredExternal<W, Store>(
  registry: ExternalBitRegistry<W, Store> | undefined,
  kind: string,
  name: string,
): ExternalBitRegistry<W, Store> {
  if (registry === undefined) {
    throw new BoolarError("external", `no external registry is installed for ${kind} '${name}'`, { kind, name });
  }
  return registry;
}

function applyAnd<W, Store>(context: BoolarContext<W, Store>, left: Value<W>, right: Value<W>): Value<W> {
  if (left.known === false || right.known === false) return { wire: context.create(false), known: false };
  if (left.known === true) return right;
  if (right.known === true) return left;
  return { wire: context.and(left.wire, right.wire), known: undefined };
}

function applyOr<W, Store>(context: BoolarContext<W, Store>, left: Value<W>, right: Value<W>): Value<W> {
  if (left.known === true || right.known === true) return { wire: context.create(true), known: true };
  if (left.known === false) return right;
  if (right.known === false) return left;
  return { wire: context.or(left.wire, right.wire), known: undefined };
}

function applyXor<W, Store>(context: BoolarContext<W, Store>, left: Value<W>, right: Value<W>): Value<W> {
  if (left.known !== undefined && right.known !== undefined) {
    const known = left.known !== right.known;
    return { wire: context.create(known), known };
  }
  if (left.known === false) return right;
  if (right.known === false) return left;
  return { wire: context.xor(left.wire, right.wire), known: undefined };
}

function validateBanks<Store>(banks: readonly StorageBank<Store>[]): Map<string, StorageBank<Store>> {
  const result = new Map<string, StorageBank<Store>>();
  for (const bank of banks) {
    if (!Number.isSafeInteger(bank.addressBits) || bank.addressBits < 0) {
      throw new BoolarError("storage", "storage bank has an invalid address width", { bank });
    }
    const key = bankKey(bank.storage, bank.lane);
    if (result.has(key)) throw new BoolarError("storage", "duplicate Boolar storage bank", { storage: bank.storage, lane: bank.lane });
    result.set(key, bank);
  }
  return result;
}

function validateCircuitStorage<Store>(circuit: BoolarCircuit, banks: ReadonlyMap<string, StorageBank<Store>>): void {
  for (const statement of circuit.statements) {
    if (statement.kind !== "storage_read" && statement.kind !== "storage_write" && statement.kind !== "action_store_bit") continue;
    const bank = banks.get(bankKey(statement.storage, statement.lane));
    if (bank === undefined) throw new BoolarError("storage", "Boolar circuit requires a missing storage bank", { storage: statement.storage, lane: statement.lane });
    assertAddressWidth(bank, statement.address.length);
  }
  for (const segment of circuit.preInit) {
    const bank = banks.get(bankKey(segment.storage, segment.lane));
    if (bank === undefined) throw new BoolarError("storage", "Boolar circuit initializes a missing storage bank", { storage: segment.storage, lane: segment.lane });
    for (let offset = 0; offset < segment.data.length; offset++) {
      if (addressBitsRequired(addStaticAddress(segment.address, offset)) > bank.addressBits) {
        throw new BoolarError("storage", "pre-initialized storage does not fit its bank", { storage: segment.storage, lane: segment.lane, offset });
      }
    }
  }
}

function initializeStorage<W, Store>(
  segments: readonly PreInitSegment[],
  context: BoolarContext<W, Store>,
  banks: ReadonlyMap<string, StorageBank<Store>>,
): void {
  for (const segment of segments) {
    const bank = banks.get(bankKey(segment.storage, segment.lane))!;
    for (let offset = 0; offset < segment.data.length; offset++) {
      const staticAddress = addStaticAddress(segment.address, offset);
      const address = Array.from({ length: bank.addressBits }, (_, bit) => {
        const known = staticAddress[bit] ?? false;
        return { wire: context.create(known), known };
      });
      context.storageWrite(bank.value, address, context.create(segment.data[offset]!));
    }
  }
}

function addStaticAddress(address: readonly boolean[], offset: number): boolean[] {
  const result = [...address];
  let remaining = offset;
  let bit = 0;
  while (remaining !== 0) {
    if ((remaining & 1) !== 0) {
      let carry = true;
      let at = bit;
      while (carry) {
        if (at === result.length) result.push(false);
        const next = result[at]! !== carry;
        carry = result[at]! && carry;
        result[at] = next;
        at++;
      }
    }
    remaining = Math.floor(remaining / 2);
    bit++;
  }
  return result;
}

function addressBitsRequired(address: readonly boolean[]): number {
  for (let bit = address.length - 1; bit >= 0; bit--) if (address[bit]) return bit + 1;
  return 0;
}

function assertAddressWidth<Store>(bank: StorageBank<Store>, found: number): void {
  if (bank.addressBits !== found) {
    throw new BoolarError("storage", "Boolar storage address width does not match its bank", {
      storage: bank.storage,
      lane: bank.lane,
      expected: bank.addressBits,
      found,
    });
  }
}

function bankKey(storage: StorageId, lane: LaneId): string { return `${storage}:${lane}`; }

/** Plaintext context suitable for tests, debugging, and Volar-spec adapters. */
export const booleanContext: BoolarContext<boolean, boolean[]> = {
  create: (value) => value,
  and: (left, right) => left && right,
  or: (left, right) => left || right,
  xor: (left, right) => left !== right,
  storageRead: (store, address) => store[booleanAddress(address)] ?? (() => { throw new RangeError("Boolar storage read is out of bounds"); })(),
  storageWrite: (store, address, value) => {
    const index = booleanAddress(address);
    if (index >= store.length) throw new RangeError("Boolar storage write is out of bounds");
    store[index] = value;
  },
};

function booleanAddress(address: readonly StorageAddressBit<boolean>[]): number {
  let result = 0;
  for (let bit = 0; bit < address.length; bit++) {
    if (address[bit]!.wire) result += 2 ** bit;
  }
  return result;
}

/** Parse the stable `volar-ir v1` Boolar text grammar, including direct bit externals. */
export function parseBoolarText(text: string): BoolarBlocks {
  const parser = new TextParser(text);
  parser.header();
  parser.expectWord("boolar");
  parser.expect("colon");
  const blocks: BoolarBlock[] = [];
  while (!parser.eof()) {
    const directive = parser.word();
    if (directive === "begin_block") {
      const id = parser.u32();
      if (id !== blocks.length) parser.fail("Boolar block IDs must be sequential");
      blocks.push(parser.block());
    } else {
      parser.fail(`unknown Boolar directive '${directive}'`);
    }
  }
  // The current upstream v1 text format does not serialize `pre_init`.
  // Preserve the IR field for callers that construct Boolar programmatically,
  // but do not silently invent a v1 syntax for it here.
  return { blocks, preInit: [] };
}

/** Emit canonical Boolar text accepted by {@link parseBoolarText}. */
export function emitBoolarText(blocks: BoolarBlocks): string {
  if (blocks.preInit.length !== 0) {
    throw new BoolarError("unsupported_statement", "volar-ir v1 Boolar text cannot encode pre_init; retain it in the programmatic IR");
  }
  const lines = ["volar-ir v1", "boolar:"];
  blocks.blocks.forEach((block, index) => {
    lines.push(`begin_block ${index}`, `params ${block.params}`);
    block.statements.forEach((statement, index2) => lines.push(`v${block.params + index2} = ${emitStatement(statement)}`));
    lines.push(emitTerminator(block.terminator), "end_block");
  });
  return `${lines.join("\n")}\n`;
}

function emitStatement(statement: BoolarStatement): string {
  switch (statement.kind) {
    case "zero": case "one": return statement.kind;
    case "and": case "or": case "xor": return `${statement.kind} v${statement.left} v${statement.right}`;
    case "not": return `not v${statement.input}`;
    case "storage_read": return `storage_read storage=${statement.storage} lane=${statement.lane} addr=${emitVars(statement.address)}`;
    case "storage_write": return `storage_write storage=${statement.storage} lane=${statement.lane} src=v${statement.source} addr=${emitVars(statement.address)}`;
    case "oracle_bit": return `oracle_bit ${JSON.stringify(statement.name)} args=${emitVars(statement.args)} bit=${statement.bit} occurrence=${statement.occurrence}`;
    case "rng_bit": return `rng_bit ${JSON.stringify(statement.name)} bit=${statement.bit} occurrence=${statement.occurrence}`;
    case "action_store_bit": return `action_store_bit ${JSON.stringify(statement.name)} guard=v${statement.guard} args=${emitVars(statement.args)} fallback=v${statement.fallback} storage=${statement.storage} lane=${statement.lane} addr=${emitVars(statement.address)} bit=${statement.bit} occurrence=${statement.occurrence}`;
    case "oracle_call": return `oracle_call ${JSON.stringify(statement.name)} args=${emitVars(statement.args)} num_bits=${statement.numBits}`;
    case "oracle_projected_bit": return `oracle_bit call=v${statement.call} bit=${statement.bit}`;
    case "action_call": return `action_call ${JSON.stringify(statement.name)} guard=v${statement.guard} args=${emitVars(statement.args)} fallback=${emitVars(statement.fallback)} num_bits=${statement.numBits}`;
    case "action_bit": return `action_bit call=v${statement.call} bit=${statement.bit}`;
    case "rng": return `rng ${JSON.stringify(statement.name)}`;
  }
}

function emitTerminator(terminator: BoolarTerminator): string {
  if (terminator.kind === "jmp") return `jmp ${emitTarget(terminator.target)} args=${emitVars(terminator.target.args)}`;
  return `cond_jmp val=v${terminator.value} then=${emitTarget(terminator.thenTarget)} then_args=${emitVars(terminator.thenTarget.args)} else=${emitTarget(terminator.elseTarget)} else_args=${emitVars(terminator.elseTarget.args)}`;
}

function emitTarget(target: BoolarTarget): string { return target.block === "return" ? "return" : `block:${target.block}`; }
function emitVars(vars: readonly VarId[]): string { return `[${vars.map((value) => `v${value}`).join(", ")}]`; }

type TokenKind = "word" | "string" | "number" | "equals" | "comma" | "lbracket" | "rbracket" | "colon" | "eof";
interface Token { readonly kind: TokenKind; readonly value: string; readonly offset: number; }

class TextParser {
  private at = 0;
  private lookahead: Token | undefined;
  constructor(private readonly input: string) {}
  header(): void {
    this.skip();
    const start = this.at;
    while (this.at < this.input.length && this.input[this.at] !== "\n") this.at++;
    if (this.input.slice(start, this.at).trim() !== "volar-ir v1") this.fail("expected 'volar-ir v1' header", start);
  }
  eof(): boolean { return this.peek().kind === "eof"; }
  expectWord(value: string): void { if (this.word() !== value) this.fail(`expected '${value}'`); }
  expect(kind: TokenKind): Token { const token = this.take(); if (token.kind !== kind) this.fail(`expected ${kind}, found ${token.value || token.kind}`, token.offset); return token; }
  word(): string { const token = this.expect("word"); return token.value; }
  string(): string { return this.expect("string").value; }
  u32(): number { const token = this.expect("number"); const value = Number(token.value); if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) this.fail("expected a u32", token.offset); return value; }
  bigint(): bigint { const token = this.expect("number"); try { return BigInt(token.value); } catch { this.fail("expected an unsigned integer", token.offset); } }
  variable(): VarId { const token = this.expect("word"); const match = /^v([0-9]+)$/.exec(token.value); if (!match) this.fail("expected a variable", token.offset); const value = Number(match![1]); if (!Number.isSafeInteger(value) || value > 0xffff_ffff) this.fail("invalid variable ID", token.offset); return value; }
  key(name: string): void { this.expectWord(name); this.expect("equals"); }
  block(): BoolarBlock {
    this.expectWord("params");
    const params = this.u32();
    const statements: BoolarStatement[] = [];
    while (true) {
      const token = this.peek();
      if (token.kind !== "word") this.fail("expected Boolar statement or terminator", token.offset);
      if (/^v[0-9]+$/.test(token.value)) {
        const output = this.variable();
        if (output !== params + statements.length) this.fail("Boolar statement IDs must be sequential", token.offset);
        this.expect("equals");
        statements.push(this.statement(this.word()));
        continue;
      }
      const directive = this.word();
      const terminator = this.terminator(directive);
      this.expectWord("end_block");
      return { params, statements, terminator };
    }
  }
  statement(kind: string): BoolarStatement {
    switch (kind) {
      case "zero": case "one": return { kind };
      case "and": case "or": case "xor": return { kind, left: this.variable(), right: this.variable() };
      case "not": return { kind, input: this.variable() };
      case "storage_read": {
        this.key("storage"); const storage = this.u32(); this.key("lane"); const lane = this.u32(); this.key("addr");
        return { kind, storage, lane, address: this.vars() };
      }
      case "storage_write": {
        this.key("storage"); const storage = this.u32(); this.key("lane"); const lane = this.u32(); this.key("src"); const source = this.variable(); this.key("addr");
        return { kind, storage, lane, source, address: this.vars() };
      }
      case "oracle_call": {
        const name = this.string(); this.key("args"); const args = this.vars(); this.key("num_bits"); return { kind, name, args, numBits: this.u32() };
      }
      case "oracle_bit": {
        if (this.peek().kind === "string") {
          const name = this.string(); this.key("args"); const args = this.vars(); this.key("bit"); const bit = this.u32(); this.key("occurrence"); const occurrence = this.bigint();
          return { kind, name, args, bit, occurrence };
        }
        this.key("call"); const call = this.variable(); this.key("bit"); return { kind: "oracle_projected_bit", call, bit: this.u32() };
      }
      case "action_call": {
        const name = this.string(); this.key("guard"); const guard = this.variable(); this.key("args"); const args = this.vars(); this.key("fallback"); const fallback = this.vars(); this.key("num_bits");
        return { kind, name, guard, args, fallback, numBits: this.u32() };
      }
      case "action_bit": this.key("call"); { const call = this.variable(); this.key("bit"); return { kind, call, bit: this.u32() }; }
      case "action_store_bit": {
        const name = this.string(); this.key("guard"); const guard = this.variable(); this.key("args"); const args = this.vars(); this.key("fallback"); const fallback = this.variable(); this.key("storage"); const storage = this.u32(); this.key("lane"); const lane = this.u32(); this.key("addr"); const address = this.vars(); this.key("bit"); const bit = this.u32(); this.key("occurrence"); const occurrence = this.bigint();
        return { kind, name, guard, args, fallback, storage, lane, address, bit, occurrence };
      }
      case "rng": return { kind, name: this.string() };
      case "rng_bit": { const name = this.string(); this.key("bit"); const bit = this.u32(); this.key("occurrence"); return { kind, name, bit, occurrence: this.bigint() }; }
      default: this.fail(`unknown Boolar statement '${kind}'`);
    }
  }
  terminator(kind: string): BoolarTerminator {
    if (kind === "jmp") { const target = this.target(); this.key("args"); return { kind, target: { ...target, args: this.vars() } }; }
    if (kind === "cond_jmp") {
      this.key("val"); const value = this.variable(); this.key("then"); const then = this.target(); this.key("then_args"); const thenTarget = { ...then, args: this.vars() }; this.key("else"); const otherwise = this.target(); this.key("else_args"); const elseTarget = { ...otherwise, args: this.vars() };
      return { kind, value, thenTarget, elseTarget };
    }
    this.fail(`unknown Boolar terminator '${kind}'`);
  }
  target(): Omit<BoolarTarget, "args"> {
    const word = this.word();
    if (word === "return") return { block: "return" };
    if (word !== "block") this.fail("expected block target");
    this.expect("colon"); return { block: this.u32() };
  }
  vars(): VarId[] { this.expect("lbracket"); const values: VarId[] = []; if (this.peek().kind !== "rbracket") while (true) { values.push(this.variable()); if (this.peek().kind !== "comma") break; this.take(); } this.expect("rbracket"); return values; }
  peek(): Token { this.lookahead ??= this.lex(); return this.lookahead; }
  take(): Token { const token = this.peek(); this.lookahead = undefined; return token; }
  fail(message: string, offset = this.at): never { const before = this.input.slice(0, offset); throw new BoolarError("parse", message, { line: before.split("\n").length, column: offset - before.lastIndexOf("\n") }); }
  private lex(): Token {
    this.skip(); const offset = this.at; const char = this.input[this.at];
    if (char === undefined) return { kind: "eof", value: "", offset };
    const single: Record<string, TokenKind> = { "=": "equals", ",": "comma", "[": "lbracket", "]": "rbracket", ":": "colon" };
    if (char in single) { this.at++; return { kind: single[char]!, value: char, offset }; }
    if (char === '"') {
      const start = this.at++; let escaped = false;
      while (this.at < this.input.length) { const current = this.input[this.at++]!; if (!escaped && current === '"') { try { return { kind: "string", value: JSON.parse(this.input.slice(start, this.at)) as string, offset }; } catch { this.fail("invalid quoted string", start); } } escaped = !escaped && current === "\\"; if (current !== "\\") escaped = false; }
      this.fail("unterminated string", start);
    }
    if (/[0-9]/.test(char)) { while (/[0-9]/.test(this.input[this.at] ?? "")) this.at++; return { kind: "number", value: this.input.slice(offset, this.at), offset }; }
    if (/[A-Za-z_]/.test(char)) { this.at++; while (/[A-Za-z0-9_]/.test(this.input[this.at] ?? "")) this.at++; return { kind: "word", value: this.input.slice(offset, this.at), offset }; }
    this.fail(`unexpected character '${char}'`, offset);
  }
  private skip(): void { while (this.at < this.input.length) { const char = this.input[this.at]!; if (/\s/.test(char)) { this.at++; continue; } if (char === ";") { while (this.at < this.input.length && this.input[this.at] !== "\n") this.at++; continue; } break; } }
}
