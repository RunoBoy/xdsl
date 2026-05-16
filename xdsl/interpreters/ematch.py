from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ordered_set import OrderedSet

from xdsl.analysis.dataflow import ChangeResult, ProgramPoint
from xdsl.analysis.sparse_analysis import Lattice, SparseForwardDataFlowAnalysis
from xdsl.dialects import ematch, equivalence
from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.dialects.pdl import RangeType
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Operation, OpResult, SSAValue, Region
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@register_impls
@dataclass
class EmatchFunctions(InterpreterFunctions):
    """Interpreter functions for PDL patterns operating on e-graphs."""

    known_ops: KnownOps = field(default_factory=KnownOps)
    """Used for hashconsing operations. When new operations are created, if they are identical to an existing operation,
    the existing operation is reused instead of creating a new one."""


    eclass_union_find: DisjointSet[equivalence.AnyClassOp] = field(
        default_factory=lambda: DisjointSet[equivalence.AnyClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

    pending_rewrites: list[tuple[SymbolRefAttr, Operation, tuple[Any, ...]]] = field(
        default_factory=lambda: []
    )
    """List of pending rewrites to be executed. Each entry is a tuple of (rewriter, root, args)."""

    worklist: list[equivalence.AnyClassOp] = field(
        default_factory=list[equivalence.AnyClassOp]
    )
    """Worklist of e-classes that need to be processed for matching."""

    is_matching: bool = True
    """Keeps track whether the interpreter is currently in a matching context (as opposed to in a rewriting context).
    If it is, finalize behaves differently by backtracking."""

    analyses: list[SparseForwardDataFlowAnalysis[Lattice[Any]]] = field(
        default_factory=lambda: []
    )
    """The sparse forward analyses to be run during equality saturation.
    These must be registered with a NonPropagatingDataFlowSolver where `propagate` is False.
    This way, state propagation is handled purely by the equality saturation logic.
    """

    def modification_handler(self, op: Operation):
        """
        Keeps `known_ops` up to date.
        Whenever an operation is modified, for example when its operands are updated to a different eclass value,
        the operation is added to the hashcons `known_ops`.
        """
        if op not in self.known_ops:
            self.known_ops[op] = op

    def populate_known_ops(self, outer_op: Operation) -> None:
        """
        Populates the known_ops dictionary by traversing the module.

        Args:
            outer_op: The operation containing all operations to be added to known_ops.
        """
        # Walk through all operations in the module
        for op in outer_op.walk():
            # Skip eclasses instances
            if not isinstance(op, equivalence.AnyClassOp):
                self.known_ops[op] = op
            else:
                self.eclass_union_find.add(op)

    @impl(ematch.AddClonedEClasses)
    def run_add_cloned_eclasses(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassValsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        input_region = args[0]
        assert isinstance(input_region, Region)

        for op in input_region.walk():
            if isinstance(op, equivalence.AnyClassOp):
                self.eclass_union_find.add(op)

        return ()

    @impl(ematch.GetClassValsOp)
    def run_get_class_vals(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassValsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Take a value and return all values in its equivalence class.

        If the value is an equivalence.class result, return the operands of the class,
        otherwise return a tuple containing just the value itself.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return ((val,),)

        assert isinstance(val, SSAValue)

        if isinstance(val, OpResult):
            defining_op = val.owner
            if isinstance(defining_op, equivalence.AnyClassOp):
                return (tuple(defining_op.operands),)

        # Value is not an eclass result, return it as a single-element tuple
        return ((val,),)

    @impl(ematch.GetClassRepresentativeOp)
    def run_get_class_representative(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassRepresentativeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get one of the values in the equivalence class of v.
        Returns the first operand of the equivalence class.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return (val,)

        assert isa(val, SSAValue)

        if isinstance(val, OpResult):
            defining_op = val.owner
            if isinstance(defining_op, equivalence.AnyClassOp):
                return (defining_op.operands[0],)

        # Value is not an eclass result, return it as-is
        return (val,)

    @impl(ematch.GetClassResultOp)
    def run_get_class_result(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get the equivalence.class result corresponding to the equivalence class of v.

        If v has exactly one use and that use is a ClassOp, return the ClassOp's result.
        Otherwise return v unchanged.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return (val,)

        assert isa(val, SSAValue)

        if val.has_one_use():
            user = val.get_user_of_unique_use()
            if isinstance(user, equivalence.AnyClassOp):
                return (user.result,)

        return (val,)

    @impl(ematch.GetClassResultsOp)
    def run_get_class_results(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get the equivalence.class results corresponding to the equivalence classes
        of a range of values.
        """
        assert len(args) == 1
        vals = args[0]

        if vals is None:
            return ((),)

        results: list[SSAValue] = []
        for val in vals:
            if val is None:
                results.append(val)
            elif val.has_one_use():
                user = val.get_user_of_unique_use()
                if isinstance(user, equivalence.AnyClassOp):
                    results.append(user.result)
                else:
                    results.append(val)
            else:
                results.append(val)

        return (tuple(results),)

    def get_or_create_class(
        self, interpreter: Interpreter, val: SSAValue
    ) -> equivalence.AnyClassOp:
        """
        Get the equivalence class for a value, creating one if it doesn't exist.
        """
        if isinstance(val, OpResult):
            # If val is defined by a ClassOp, return it
            if isinstance(val.owner, equivalence.AnyClassOp):
                return val.owner
            insertpoint = InsertPoint.before(val.owner)
        else:
            assert isinstance(val.owner, Block)
            insertpoint = InsertPoint.at_start(val.owner)

        # If val has one use and it's a ClassOp, return it
        if (user := val.get_user_of_unique_use()) is not None:
            if isinstance(user, equivalence.AnyClassOp):
                return user

        # If the value is not part of an eclass yet, create one
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)

        eclass_op = equivalence.ClassOp(val)
        rewriter.insert_op(eclass_op, insertpoint)
        self.eclass_union_find.add(eclass_op)

        # Replace uses of val with the eclass result (except in the eclass itself)
        rewriter.replace_uses_with_if(
            val, eclass_op.result, lambda use: use.operation is not eclass_op
        )

        return eclass_op

    def eclass_union(
            self,
            interpreter: Interpreter,
            a: equivalence.AnyClassOp,
            b: equivalence.AnyClassOp,
    ) -> bool:
        """Unions two eclasses, ensuring the one highest in the block/region scope is preserved."""
        a = self.eclass_union_find.find(a)
        b = self.eclass_union_find.find(b)

        if a == b:
            return False

        # Meet the dataflow analysis states of the two e-classes
        for analysis in self.analyses:
            a_lattice = analysis.get_lattice_element(a.result)
            b_lattice = analysis.get_lattice_element(b.result)
            a_lattice.meet(b_lattice)

        # Helper logic to compute structural nesting depth
        def get_scope_depth(op: Operation) -> int:
            depth = 0
            curr = op.parent_op()
            while curr is not None:
                depth += 1
                curr = curr.parent_op()
            return depth

        # Choose the survivor based on type and scope rules
        if isinstance(a, equivalence.ConstantClassOp):
            if isinstance(b, equivalence.ConstantClassOp):
                assert a.value == b.value, "Trying to union two different constant eclasses."
            to_keep, to_replace = a, b
            self.eclass_union_find.union_left(to_keep, to_replace)
        elif isinstance(b, equivalence.ConstantClassOp):
            to_keep, to_replace = b, a
            self.eclass_union_find.union_left(to_keep, to_replace)
        else:
            # FIX: Compare lexical nesting depths. Lower depth = higher in structural scope.
            depth_a = get_scope_depth(a)
            depth_b = get_scope_depth(b)

            if depth_a == 0:
                to_keep, to_replace = b, a
            elif depth_b == 0:
                to_keep, to_replace = a, b
            else:
                if depth_a <= depth_b:
                    to_keep, to_replace = a, b
                else:
                    to_keep, to_replace = b, a

            # Use union_left to strictly preserve the chosen representative
            self.eclass_union_find.union_left(to_keep, to_replace)

        # Merge operands and clean
        new_operands = OrderedSet(to_keep.operands)
        new_operands.update(to_replace.operands)

        clean_operands = OrderedSet()
        for op in new_operands:
            if isinstance(op, OpResult) and isinstance(op.owner, equivalence.AnyClassOp):
                continue
            clean_operands.add(op)

        to_keep.operands = tuple(clean_operands)

        for use in to_replace.result.uses:
            if use.operation in self.known_ops:
                self.known_ops.pop(use.operation)

        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        rewriter.replace_op(to_replace, new_ops=[], new_results=to_keep.results)
        return True

    def union_val(self, interpreter: Interpreter, a: SSAValue, b: SSAValue) -> None:
        """
        Union two values into the same equivalence class.
        """
        if a == b:
            return

        eclass_a = self.get_or_create_class(interpreter, a)
        eclass_b = self.get_or_create_class(interpreter, b)

        if self.eclass_union(interpreter, eclass_a, eclass_b):
            self.worklist.append(eclass_a)

    @impl(ematch.UnionOp)
    def run_union(
        self,
        interpreter: Interpreter,
        op: ematch.UnionOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Merge two values, an operation and a value range, or two value ranges
        into equivalence class(es).

        Supported operand type combinations:
        - (value, value): merge two values
        - (operation, range<value>): merge operation results with values
        - (range<value>, range<value>): merge two value ranges
        """
        assert len(args) == 2
        lhs, rhs = args

        if isa(lhs, SSAValue) and isa(rhs, SSAValue):
            # (Value, Value) case
            self.union_val(interpreter, lhs, rhs)

        elif isinstance(lhs, Operation) and isa(rhs, Sequence[SSAValue]):
            # (Operation, ValueRange) case
            assert len(lhs.results) == len(rhs), (
                "Operation result count must match value range size"
            )
            for result, val in zip(lhs.results, rhs, strict=True):
                self.union_val(interpreter, result, val)

        elif isa(lhs, Sequence[SSAValue]) and isa(rhs, Sequence[SSAValue]):
            # (ValueRange, ValueRange) case
            assert len(lhs) == len(rhs), "Value ranges must have equal size"
            for val_lhs, val_rhs in zip(lhs, rhs, strict=True):
                self.union_val(interpreter, val_lhs, val_rhs)

        else:
            raise InterpretationError(
                f"union: unsupported argument types: {type(lhs)}, {type(rhs)}"
            )

        return ()

    @impl(ematch.DedupRegionOp)
    def run_dedup_region(
            self,
            interpreter: Interpreter,
            op: ematch.DedupRegionOp,
            args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        inlined_ops = args[0]
        ops_added = []
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)

        # PROTECT TERMINATORS (Make sure this is imported!)
        from xdsl.traits import IsTerminator

        for input_op in inlined_ops:
            if input_op.parent is None:
                continue

            # --- PHASE 1: Handle E-Classes ---
            if isinstance(input_op, equivalence.AnyClassOp):
                try:
                    self.eclass_union_find.find(input_op)
                except KeyError:
                    self.eclass_union_find.add(input_op)

                merged = False
                for operand in input_op.operands:
                    for use in list(operand.uses):
                        if use.operation is not input_op and isinstance(use.operation, equivalence.AnyClassOp):
                            self.eclass_union(interpreter, input_op, use.operation)
                            merged = True
                            break
                    if merged:
                        break

                if not merged:
                    for use in list(input_op.result.uses):
                        if isinstance(use.operation, equivalence.AnyClassOp) and use.operation is not input_op:
                            self.eclass_union(interpreter, input_op, use.operation)
                            break
                continue

            # --- PHASE 2: Handle Standard Operations ---
            if input_op.has_trait(IsTerminator):
                continue

            existing = self.known_ops.get(input_op)

            if existing is not None and existing is not input_op:
                for res_old, res_new in zip(input_op.results, existing.results):
                    self.union_val(interpreter, res_old, res_new)

                rewriter.replace_op(input_op, new_ops=[], new_results=existing.results)

                for res_new in existing.results:
                    for use in list(res_new.uses):
                        if isinstance(use.operation, equivalence.AnyClassOp):
                            unique_ops = list(dict.fromkeys(use.operation.operands))
                            use.operation.operands = tuple(unique_ops)
            else:
                self.known_ops[input_op] = input_op
                ops_added.append(input_op)

        # EXACTLY 0 means all computations were perfectly deduplicated
        if len(ops_added) == 0:
            return (None,)
        return (ops_added,)

    @impl(ematch.DedupOp)
    def run_dedup(
        self,
        interpreter: Interpreter,
        op: ematch.DedupOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Check if the operation already exists in the hashcons.

        If an equivalent operation exists, erase the input operation and return
        the existing one. Otherwise, insert the operation into the hashcons and
        return it.
        """
        assert len(args) == 1
        input_op = args[0]
        assert isinstance(input_op, Operation)

        # Check if an equivalent operation exists in hashcons
        existing = self.known_ops.get(input_op)

        if existing is not None and existing is not input_op:
            # Deduplicate: erase the new op and return existing
            rewriter = PDLInterpFunctions.get_rewriter(interpreter)
            rewriter.erase_op(input_op)
            return (existing,)

        # No duplicate found, insert into hashcons
        self.known_ops[input_op] = input_op
        return (input_op,)


    def repair(self, interpreter: Interpreter, eclass: equivalence.AnyClassOp):
        """
        Repair an e-class by finding and merging duplicate parent operations.

        This method:
        1. Finds all operations that use this e-class's result
        2. Identifies structurally equivalent operations among them
        3. Merges equivalent operations by unioning their result e-classes
        4. Updates dataflow analysis states
        """
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        eclass = self.eclass_union_find.find(eclass)

        if eclass.parent is None:
            return

            # --- NEW: E-Class Overlap Detection ---
            # Look at the operands inside this e-class. If another e-class is tracking
            # the exact same operand, they are duplicates and must be merged!
        for operand in list(eclass.operands):
            for use in list(operand.uses):
                other_op = use.operation
                if isinstance(other_op, equivalence.AnyClassOp) and other_op is not eclass:
                    if self.eclass_union(interpreter, eclass, other_op):
                        # Update our reference to the surviving e-class
                        eclass = self.eclass_union_find.find(eclass)
                        # Add the surviving e-class to the worklist for further repair
                        self.worklist.append(eclass)

            # If this specific e-class instance was erased during the merge, stop processing it
        if eclass.parent is None:
            return

        unique_parents = KnownOps()

        # Collect e-class parents (operations that use this class's result)
        # Use OrderedSet to maintain deterministic ordering
        user_ops = OrderedSet(use.operation for use in eclass.result.uses)

        # Collect pairs of duplicate operations to merge AFTER the loop
        # This avoids modifying the hash map while iterating
        to_merge: list[tuple[Operation, Operation]] = []

        for op1 in user_ops:
            # Skip eclass operations themselves
            if isinstance(op1, equivalence.AnyClassOp):
                continue

            op2 = unique_parents.get(op1)

            if op2 is not None:
                # Found an equivalent operation - record for later merging
                to_merge.append((op1, op2))
            else:
                unique_parents[op1] = op1

        # Now perform all merges after we're done with the hash map
        for op1, op2 in to_merge:
            # Collect eclass pairs for ALL results before replacement
            eclass_pairs: list[
                tuple[equivalence.AnyClassOp, equivalence.AnyClassOp]
            ] = []
            for res1, res2 in zip(op1.results, op2.results, strict=True):
                eclass1 = self.get_or_create_class(interpreter, res1)
                eclass2 = self.get_or_create_class(interpreter, res2)
                eclass_pairs.append((eclass1, eclass2))

            # Replace op1 with op2's results
            rewriter.replace_op(op1, new_ops=(), new_results=op2.results)
            self.known_ops[op2] = op2

            # Process each eclass pair
            for eclass1, eclass2 in eclass_pairs:
                if eclass1 == eclass2:
                    # Same eclass - just deduplicate operands
                    eclass1.operands = OrderedSet(eclass1.operands)
                else:
                    # Different eclasses - union them
                    if self.eclass_union(interpreter, eclass1, eclass2):
                        self.worklist.append(eclass1)

        # Update dataflow analysis for all parent operations
        eclass = self.eclass_union_find.find(eclass)
        for op in OrderedSet(use.operation for use in eclass.result.uses):
            if isinstance(op, equivalence.AnyClassOp):
                continue

            point = ProgramPoint.before(op)

            for analysis in self.analyses:
                operands = [
                    analysis.get_lattice_element_for(point, o) for o in op.operands
                ]
                results = [analysis.get_lattice_element(r) for r in op.results]

                if not results:
                    continue

                original_state: Any = None
                # For each result, reset to bottom and recompute
                for result in results:
                    original_state = result.value
                    result._value = result.value_cls()  # pyright: ignore[reportPrivateUsage]

                analysis.visit_operation_impl(op, operands, results)

                # Check if any result changed
                for result in results:
                    assert original_state is not None
                    changed = result.meet(type(result)(result.anchor, original_state))
                    if changed == ChangeResult.CHANGE:
                        # Find the eclass for this result and add to worklist
                        if (op_use := op.results[0].first_use) is not None:
                            if isinstance(
                                eclass_op := op_use.operation, equivalence.AnyClassOp
                            ):
                                self.worklist.append(eclass_op)
                        break  # Only need to add to worklist once per operation

    def rebuild(self, interpreter: Interpreter):
        while self.worklist:
            todo = OrderedSet(self.eclass_union_find.find(c) for c in self.worklist)
            self.worklist.clear()
            for c in todo:
                self.repair(interpreter, c)

    def execute_pending_rewrites(self, interpreter: Interpreter):
        """Execute all pending rewrites that were aggregated during matching."""
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        for rewriter_op, root, args in self.pending_rewrites:
            rewriter.current_operation = root
            rewriter.insertion_point = InsertPoint.before(root)

            self.is_matching = False
            interpreter.call_op(rewriter_op, args)
            self.is_matching = True
        self.pending_rewrites.clear()
