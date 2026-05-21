// RUN: xdsl-opt %s -p ematch-saturate | filecheck %s

// CHECK:func.func @main(%cond: i1, %a: f32, %b: f32) -> f32 {
// CHECK-NEXT:    %res = equivalence.graph : () -> f32 {
// CHECK-NEXT:      %sum_comm = equivalence.class %sum, %sum_comm_1 : f32
// CHECK-NEXT:      %sum = arith.addf %a, %b : f32
// CHECK-NEXT:      %sum_comm_1 = arith.addf %b, %a : f32
// CHECK-NEXT:      %if = scf.if %cond -> (f32) {
// CHECK-NEXT:        %x = func.call @func_a(%sum_comm) : (f32) -> f32
// CHECK-NEXT:        scf.yield %x : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %y = func.call @func_b(%sum_comm) : (f32) -> f32
// CHECK-NEXT:        scf.yield %y : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      equivalence.yield %if : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %res : f32
// CHECK-NEXT:  }

func.func private @func_a(f32) -> f32
func.func private @func_b(f32) -> f32

func.func @main(%cond : i1, %a : f32, %b : f32) -> f32 {
    %res = equivalence.graph : () -> f32 {
        %if = scf.if %cond -> (f32) {
            %sum = arith.addf %a, %b : f32
            // Corrected call syntax
            %x = func.call @func_a(%sum) : (f32) -> f32
            scf.yield %x : f32
        } else {
            %sum_comm = arith.addf %b, %a : f32
            // Corrected call syntax
            %y = func.call @func_b(%sum_comm) : (f32) -> f32
            scf.yield %y : f32
        }
        equivalence.yield %if : f32
    }
    func.return %res : f32
}


pdl_interp.func @matcher(%arg0: !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "arith.addf" -> ^bb_addf, ^bb_if_collapse
  ^bb_fail:
    pdl_interp.finalize
  ^bb_addf:
    pdl_interp.record_match @rewriters::@addf_comm(%arg0 : !pdl.operation) : benefit(2), loc([]) -> ^bb_fail
  ^bb_if_collapse:
    pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb_if_regions, ^bb_fail
  ^bb_if_regions:
    %r0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
    %r1 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
    pdl_interp.record_match @rewriters::@if_collapse(%arg0, %r0, %r1 : !pdl.operation, !pdl_region.region, !pdl_region.region) : benefit(3), loc([]) -> ^bb_fail
}

module @rewriters {
    pdl_interp.func @addf_comm(%arg0 : !pdl.operation) {
        %a = pdl_interp.get_operand 0 of %arg0
        %b = pdl_interp.get_operand 1 of %arg0
        %d = pdl_interp.get_result 0 of %arg0
        %e = pdl_interp.get_value_type of %d : !pdl.type
        %c = pdl_interp.create_operation "arith.addf"(%b, %a : !pdl.value, !pdl.value) -> (%e : !pdl.type)
        %i = ematch.dedup %c

        %f = pdl_interp.get_result 0 of %i
        %g = ematch.get_class_result %f
        %h = pdl_interp.create_range %g : !pdl.value
        ematch.union %arg0 : !pdl.operation, %h : !pdl.range<value>
        pdl_interp.finalize
    }

    pdl_interp.func @if_collapse(%arg0: !pdl.operation, %r0 : !pdl_region.region, %r1 : !pdl_region.region) {
      %y0 = pdl_interp_region.get_operation() called "scf.yield" 0 of %r0
      %y1 = pdl_interp_region.get_operation() called "scf.yield" 0 of %r1
      %v0 = pdl_interp.get_operand 0 of %y0
      %v1 = pdl_interp.get_operand 0 of %y1

      %e0 = ematch.get_class_result %v0
      %e1 = ematch.get_class_result %v1

      pdl_interp.are_equal %e0, %e1 : !pdl.value -> ^bb_match, ^bb_fail

    ^bb_match:
      pdl_interp.replace %arg0 with (%v0 : !pdl.value)
      pdl_interp.finalize

    ^bb_fail:
      pdl_interp.finalize
    }
}
