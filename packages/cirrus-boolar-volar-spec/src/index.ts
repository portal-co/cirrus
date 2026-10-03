/**
 * Bridge concrete Boolar external bits to functions generated from volar-spec.
 *
 * Boolar carries a source name and flattened bits, not source-level type
 * metadata.  Callers therefore register codecs explicitly; this avoids any
 * lossy or reflection-based guess about a generated TypeScript signature.
 */
import type {
  ActionStoreBit,
  BoolarContext,
  ExternalBitRegistry,
  StorageAddressBit,
  StorageBank,
} from "@portal-solutions/cirrus-boolar";

export interface BitCodec<T> {
  readonly width: number;
  decode(bits: readonly boolean[]): T;
  encode(value: T): readonly boolean[];
}

export interface VolarOracle<Input, Output> {
  readonly kind: "oracle";
  readonly name: string;
  readonly input: BitCodec<Input>;
  readonly output: BitCodec<Output>;
  invoke(input: Input): Output;
}

export interface VolarAction<Input, Output> {
  readonly kind: "action";
  readonly name: string;
  readonly input: BitCodec<Input>;
  readonly output: BitCodec<Output>;
  invoke(input: Input): Output;
}

export interface BitRandomSource {
  nextBit(bit: number, occurrence: bigint): boolean;
}

export interface VolarRng {
  readonly kind: "rng";
  readonly name: string;
  readonly source: BitRandomSource;
}

export type VolarOperation = VolarOracle<unknown, unknown> | VolarAction<unknown, unknown> | VolarRng;

/**
 * Create a registry for the Boolean debugging/execution host.
 *
 * An action is invoked once for each `ActionStoreBit`, exactly matching the
 * direct Boolar operation.  Do not register a multi-output side effect here
 * unless its bit-level invocation is independently safe.
 */
export function createVolarSpecRegistry<Store>(
  operations: readonly VolarOperation[],
): ExternalBitRegistry<boolean, Store> {
  const oracles = new Map<string, VolarOracle<unknown, unknown>>();
  const actions = new Map<string, VolarAction<unknown, unknown>>();
  const rngs = new Map<string, VolarRng>();
  for (const operation of operations) {
    const map = operation.kind === "oracle" ? oracles : operation.kind === "action" ? actions : rngs;
    if (map.has(operation.name)) throw new Error(`duplicate Volar operation '${operation.name}'`);
    map.set(operation.name, operation as never);
  }
  return {
    oracleBit(_context, name, args, bit) {
      const operation = oracles.get(name);
      if (operation === undefined) throw new Error(`unregistered Volar oracle '${name}'`);
      const input = decode(operation.input, args, name, "input");
      return selectBit(operation.output.encode(operation.invoke(input)), operation.output.width, bit, name);
    },
    rngBit(_context, name, bit, occurrence) {
      const operation = rngs.get(name);
      if (operation === undefined) throw new Error(`unregistered Volar RNG '${name}'`);
      return operation.source.nextBit(bit, occurrence);
    },
    actionStoreBit(context, call, guard, args, fallback, address, bank) {
      const operation = actions.get(call.name);
      if (operation === undefined) throw new Error(`unregistered Volar action '${call.name}'`);
      const value = guard
        ? selectBit(operation.output.encode(operation.invoke(decode(operation.input, args, call.name, "input"))), operation.output.width, call.bit, call.name)
        : fallback;
      context.storageWrite(bank.value, address, value);
    },
  };
}

function decode<T>(codec: BitCodec<T>, bits: readonly boolean[], name: string, position: string): T {
  if (bits.length !== codec.width) throw new Error(`${name} ${position} expects ${codec.width} bits, got ${bits.length}`);
  return codec.decode(bits);
}

function selectBit(bits: readonly boolean[], width: number, bit: number, name: string): boolean {
  if (bits.length !== width) throw new Error(`${name} result codec promised ${width} bits, produced ${bits.length}`);
  const value = bits[bit];
  if (value === undefined) throw new Error(`${name} does not produce result bit ${bit}`);
  return value;
}

/** Reusable little-endian unsigned codec for generated functions using bigint. */
export function unsignedBits(width: number): BitCodec<bigint> {
  if (!Number.isSafeInteger(width) || width < 0) throw new RangeError("invalid bit width");
  return {
    width,
    decode(bits) { return bits.reduce((value, bit, index) => bit ? value | (1n << BigInt(index)) : value, 0n); },
    encode(value) { return Array.from({ length: width }, (_, bit) => (value & (1n << BigInt(bit))) !== 0n); },
  };
}

export function booleanBit(): BitCodec<boolean> {
  return {
    width: 1,
    decode(bits) { if (bits.length !== 1) throw new Error("Boolean requires one bit"); return bits[0]!; },
    encode(value) { return [value]; },
  };
}
