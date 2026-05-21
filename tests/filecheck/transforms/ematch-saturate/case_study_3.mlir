// RUN: xdsl-opt %s -p ematch-saturate | filecheck %s

// CHECK:func.func @main(%cond: i1, %cond2: i1, %a: f32) -> (f32, f32) {
// CHECK-NEXT:    %r1, %r2 = equivalence.graph : () -> (f32, f32) {
// CHECK-NEXT:      %val1 = equivalence.class %val1_1 : f32
// CHECK-NEXT:      %val1_1 = func.call @func_a(%a) : (f32) -> f32
// CHECK-NEXT:      %res1 = scf.if %cond -> (f32) {
// CHECK-NEXT:        scf.yield %val1 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %a : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %res2 = scf.if %cond2 -> (f32) {
// CHECK-NEXT:        scf.yield %val1 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %a : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      equivalence.yield %res1, %res2 : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %r1, %r2 : f32, f32
// CHECK-NEXT:  }

func.func private @func_a(f32) -> f32

func.func @main(%cond : i1, %cond2 : i1, %a : f32) -> (f32, f32) {
    %r1, %r2 = equivalence.graph : () -> (f32, f32) {
        %res1 = scf.if %cond -> (f32) {
            %val1 = func.call @func_a(%a) : (f32) -> f32
            scf.yield %val1 : f32
        } else {
            scf.yield %a : f32
        }

        %res2 = scf.if %cond2 -> (f32) {
            %val2 = func.call @func_a(%a) : (f32) -> f32
            scf.yield %val2 : f32
        } else {
            scf.yield %a : f32
        }

        equivalence.yield %res1, %res2 : f32, f32
    }
    func.return %r1, %r2 : f32, f32
}


pdl_interp.func @matcher(%arg0: !pdl.operation) {
   pdl_interp.finalize
}
