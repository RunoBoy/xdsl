// CHECK:  func.func @log_return(%arg0: f32) -> f32 {
// CHECK-NEXT:    %0 = equivalence.graph : () -> f32 {
// CHECK-NEXT:      %1 = math.log %arg0 : f32
// CHECK-NEXT:      equivalence.yield %1 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @log_return2(%arg0: f32) -> f32 {
// CHECK-NEXT:    %0 = equivalence.graph : () -> f32 {
// CHECK-NEXT:      %1 = math.log %arg0 : f32
// CHECK-NEXT:      %2 = equivalence.class %3, %1, %4 : f32
// CHECK-NEXT:      %4 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %5 = math.log %arg0 : f32
// CHECK-NEXT:        scf.yield %5 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %3 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:      equivalence.yield %2 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @log_return3(%arg0: f32) -> f32 {
// CHECK-NEXT:    %0 = equivalence.graph : () -> f32 {
// CHECK-NEXT:      %1 = math.log %arg0 : f32
// CHECK-NEXT:      %2 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %3 = math.log %arg0 : f32
// CHECK-NEXT:        scf.yield %3 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:      %5 = equivalence.class %6, %4, %1, %2, %7 : f32
// CHECK-NEXT:      %7 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %8 = math.log %arg0 : f32
// CHECK-NEXT:        %9 = equivalence.class %10, %8, %11 : f32
// CHECK-NEXT:        %11 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %12 = math.log %arg0 : f32
// CHECK-NEXT:          scf.yield %12 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %10 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:        scf.yield %9 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = func.call @log_return2(%arg0) : (f32) -> f32
// CHECK-NEXT:      equivalence.yield %5 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @compound(%arg0: f32) -> f32 {
// CHECK-NEXT:    %0 = equivalence.graph : () -> f32 {
// CHECK-NEXT:      %1 = math.exp %arg0 : f32
// CHECK-NEXT:      equivalence.yield %1 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @quant_model(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = equivalence.graph : () -> f32 {
// CHECK-NEXT:      %1 = math.log %arg0 : f32
// CHECK-NEXT:      %2 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %3 = math.log %arg0 : f32
// CHECK-NEXT:        scf.yield %3 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:      %5 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %6 = math.log %arg0 : f32
// CHECK-NEXT:        %7 = equivalence.class %8, %6, %9 : f32
// CHECK-NEXT:        %9 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %10 = math.log %arg0 : f32
// CHECK-NEXT:          scf.yield %10 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %8 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:        scf.yield %7 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %11 = func.call @log_return2(%arg0) : (f32) -> f32
// CHECK-NEXT:      %12 = equivalence.class %13, %11, %4, %1, %2, %5, %14 : f32
// CHECK-NEXT:      %14 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %15 = math.log %arg0 : f32
// CHECK-NEXT:        %16 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %17 = math.log %arg0 : f32
// CHECK-NEXT:          scf.yield %17 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %18 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:        %19 = equivalence.class %20, %18, %15, %16, %21 : f32
// CHECK-NEXT:        %21 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %22 = math.log %arg0 : f32
// CHECK-NEXT:          %23 = equivalence.class %24, %22, %25 : f32
// CHECK-NEXT:          %25 = scf.execute_region -> (f32) {
// CHECK-NEXT:            %26 = math.log %arg0 : f32
// CHECK-NEXT:            scf.yield %26 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          %24 = func.call @log_return(%arg0) : (f32) -> f32
// CHECK-NEXT:          scf.yield %23 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %20 = func.call @log_return2(%arg0) : (f32) -> f32
// CHECK-NEXT:        scf.yield %19 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %13 = func.call @log_return3(%arg0) : (f32) -> f32
// CHECK-NEXT:      %27 = math.log %arg1 : f32
// CHECK-NEXT:      %28 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %29 = math.log %arg1 : f32
// CHECK-NEXT:        scf.yield %29 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %30 = func.call @log_return(%arg1) : (f32) -> f32
// CHECK-NEXT:      %31 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %32 = math.log %arg1 : f32
// CHECK-NEXT:        %33 = equivalence.class %34, %32, %35 : f32
// CHECK-NEXT:        %35 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %36 = math.log %arg1 : f32
// CHECK-NEXT:          scf.yield %36 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %34 = func.call @log_return(%arg1) : (f32) -> f32
// CHECK-NEXT:        scf.yield %33 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %37 = func.call @log_return2(%arg1) : (f32) -> f32
// CHECK-NEXT:      %38 = equivalence.class %39, %37, %30, %27, %28, %31, %40 : f32
// CHECK-NEXT:      %40 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %41 = math.log %arg1 : f32
// CHECK-NEXT:        %42 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %43 = math.log %arg1 : f32
// CHECK-NEXT:          scf.yield %43 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %44 = func.call @log_return(%arg1) : (f32) -> f32
// CHECK-NEXT:        %45 = equivalence.class %46, %44, %41, %42, %47 : f32
// CHECK-NEXT:        %47 = scf.execute_region -> (f32) {
// CHECK-NEXT:          %48 = math.log %arg1 : f32
// CHECK-NEXT:          %49 = equivalence.class %50, %48, %51 : f32
// CHECK-NEXT:          %51 = scf.execute_region -> (f32) {
// CHECK-NEXT:            %52 = math.log %arg1 : f32
// CHECK-NEXT:            scf.yield %52 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          %50 = func.call @log_return(%arg1) : (f32) -> f32
// CHECK-NEXT:          scf.yield %49 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %46 = func.call @log_return2(%arg1) : (f32) -> f32
// CHECK-NEXT:        scf.yield %45 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %39 = func.call @log_return3(%arg1) : (f32) -> f32
// CHECK-NEXT:      %arg0_1 = arith.addf %12, %38 : f32
// CHECK-NEXT:      %53 = arith.mulf %arg0, %arg1 : f32
// CHECK-NEXT:      %54 = math.exp %arg0_1 : f32
// CHECK-NEXT:      %55 = equivalence.class %56, %54, %53, %57 : f32
// CHECK-NEXT:      %57 = scf.execute_region -> (f32) {
// CHECK-NEXT:        %58 = math.exp %arg0_1 : f32
// CHECK-NEXT:        scf.yield %58 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %56 = func.call @compound(%arg0_1) : (f32) -> f32
// CHECK-NEXT:      equivalence.yield %55 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %0 : f32
// CHECK-NEXT:  }