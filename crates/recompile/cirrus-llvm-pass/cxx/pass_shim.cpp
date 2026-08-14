#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm-c/Core.h"

extern "C" int cirrus_llvm_pass_run(LLVMModuleRef Module, char **Error);
extern "C" void cirrus_llvm_pass_free_error(char *Error);

// Referenced from Rust so the archive member that contains
// llvmGetPassPluginInfo is retained when Cargo links this cdylib.
extern "C" void cirrus_llvm_pass_link_anchor() {}

namespace {

class CirrusLowerPass : public llvm::PassInfoMixin<CirrusLowerPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
    char *Error = nullptr;
    int Result = cirrus_llvm_pass_run(reinterpret_cast<LLVMModuleRef>(&M), &Error);
    if (Result < 0) {
      std::string Message = Error ? Error : "unknown Cirrus pass failure";
      if (Error)
        cirrus_llvm_pass_free_error(Error);
      Message = "cirrus-lower: " + Message;
      llvm::report_fatal_error(llvm::StringRef(Message));
    }
    return Result == 0 ? llvm::PreservedAnalyses::all()
                       : llvm::PreservedAnalyses::none();
  }
};

void registerCallbacks(llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
         llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name != "cirrus-lower")
          return false;
        MPM.addPass(CirrusLowerPass());
        return true;
      });
  PB.registerOptimizerEarlyEPCallback(
      [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel,
         llvm::ThinOrFullLTOPhase Phase) {
        // Full-LTO pre-link sees individual modules.  Its merged-module hook
        // below is the only phase allowed to resolve cross-module selectors.
        if (Phase != llvm::ThinOrFullLTOPhase::FullLTOPreLink)
          MPM.addPass(CirrusLowerPass());
      });
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(
      [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
        MPM.addPass(CirrusLowerPass());
      });
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "cirrus-llvm-pass", "0.1", registerCallbacks};
}
