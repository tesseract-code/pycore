"""
Advanced Signature Type Mapping Interface

This module provides comprehensive type analysis and mapping for function and class signatures.
It extracts parameter names and their types, with intelligent type reduction to find base types.
"""

import inspect
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from enum import unique, StrEnum
from typing import (
    Any, Dict, List, Tuple, Set, Union, Optional, Callable, Type,
    get_origin, get_args, get_type_hints, ForwardRef
)


@unique
class TypeComplexity(StrEnum):
    """Enumeration of type complexity levels."""
    PRIMITIVE = "primitive"  # int, str, bool, float, etc.
    BUILTIN_COLLECTION = "builtin_collection"  # list, dict, tuple, set
    TYPING_GENERIC = "typing_generic"  # List[T], Dict[K,V], Optional[T]
    UNION = "union"  # Union[A, B], A | B
    CALLABLE = "callable"  # Callable[[...], ...]
    CLASS = "class"  # User-defined classes
    COMPLEX = "complex"  # Complex nested types
    UNKNOWN = "unknown"  # Can't determine type


@dataclass
class TypeInfo:
    """Comprehensive information about a parameter type."""
    name: str
    original_type: Any
    base_type: Type
    complexity: TypeComplexity
    type_args: List[Any] = field(default_factory=list)
    origin_type: Optional[Type] = None
    is_optional: bool = False
    default_value: Any = inspect.Parameter.empty
    annotation_string: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TypeReducer:
    """
    Advanced type analysis and reduction engine.

    Analyzes complex types and reduces them to their most fundamental components.
    """

    # Primitive types that can't be reduced further
    PRIMITIVE_TYPES = {
        int, float, str, bool, bytes, bytearray,
        complex, type(None), type(Ellipsis), type(NotImplemented)
    }

    # Built-in collection types
    BUILTIN_COLLECTIONS = {
        list, dict, tuple, set, frozenset, range,
        enumerate, zip, filter, map
    }

    @classmethod
    def analyze_type(cls, type_hint: Any, param_name: str = "") -> TypeInfo:
        """
        Analyze a type hint and extract comprehensive type information.

        Args:
            type_hint: The type hint to analyze
            param_name: Name of the parameter (for context)

        Returns:
            TypeInfo object with detailed type analysis
        """
        if type_hint is inspect.Parameter.empty:
            return TypeInfo(
                name=param_name,
                original_type=type_hint,
                base_type=Any,
                complexity=TypeComplexity.UNKNOWN,
                annotation_string="Any"
            )

        # Handle string annotations (forward references)
        if isinstance(type_hint, str):
            return cls._analyze_string_type(type_hint, param_name)

        # Handle ForwardRef
        if isinstance(type_hint, ForwardRef):
            return cls._analyze_forward_ref(type_hint, param_name)

        # Get origin and args for generic types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Determine complexity and base type
        complexity = cls._determine_complexity(type_hint, origin, args)
        base_type = cls._extract_base_type(type_hint, origin, args, complexity)

        # Check if optional
        is_optional = cls._is_optional_type(type_hint, origin, args)

        return TypeInfo(
            name=param_name,
            original_type=type_hint,
            base_type=base_type,
            complexity=complexity,
            type_args=list(args) if args else [],
            origin_type=origin,
            is_optional=is_optional,
            annotation_string=cls._format_type_string(type_hint)
        )

    @classmethod
    def _analyze_string_type(cls, type_str: str, param_name: str) -> TypeInfo:
        """Analyze string type annotations."""
        # Try to evaluate the string in common contexts
        try:
            # Common type evaluation contexts
            type_globals = {
                **typing.__dict__,
                'Any': Any, 'Union': Union, 'Optional': Optional,
                'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Set': Set,
                'Callable': Callable, 'Type': Type
            }

            # Attempt to evaluate
            evaluated_type = eval(type_str, type_globals)
            return cls.analyze_type(evaluated_type, param_name)

        except (NameError, SyntaxError, TypeError):
            # If evaluation fails, treat as unknown
            return TypeInfo(
                name=param_name,
                original_type=type_str,
                base_type=Any,
                complexity=TypeComplexity.UNKNOWN,
                annotation_string=type_str
            )

    @classmethod
    def _analyze_forward_ref(cls, forward_ref: ForwardRef,
                             param_name: str) -> TypeInfo:
        """Analyze ForwardRef type annotations."""
        try:
            # Try to resolve the forward reference
            if hasattr(forward_ref, '__forward_arg__'):
                type_str = forward_ref.__forward_arg__
            else:
                type_str = str(forward_ref)

            return cls._analyze_string_type(type_str, param_name)

        except Exception:
            return TypeInfo(
                name=param_name,
                original_type=forward_ref,
                base_type=Any,
                complexity=TypeComplexity.UNKNOWN,
                annotation_string=str(forward_ref)
            )

    @classmethod
    def _determine_complexity(cls, type_hint: Any, origin: Any,
                              args: Tuple) -> TypeComplexity:
        """Determine the complexity level of a type."""

        # Check for primitive types first
        if type_hint in cls.PRIMITIVE_TYPES:
            return TypeComplexity.PRIMITIVE

        # Handle None type specially
        if type_hint is type(None):
            return TypeComplexity.PRIMITIVE

        # Check for built-in collections
        if origin in cls.BUILTIN_COLLECTIONS or type_hint in cls.BUILTIN_COLLECTIONS:
            return TypeComplexity.BUILTIN_COLLECTION

        # Check for Union types (including Optional)
        if origin is Union:
            return TypeComplexity.UNION

        # Check for Callable types
        if origin is typing.Callable or (hasattr(typing, 'collections') and
                                         origin is typing.collections.abc.Callable):
            return TypeComplexity.CALLABLE

        # Check for generic types from typing module
        if origin is not None:
            if hasattr(typing,
                       origin.__name__ if hasattr(origin, '__name__') else ''):
                return TypeComplexity.TYPING_GENERIC
            return TypeComplexity.TYPING_GENERIC

        # Check if it's a class
        if inspect.isclass(type_hint):
            return TypeComplexity.CLASS

        # If we have type arguments, it's likely complex
        if args:
            return TypeComplexity.COMPLEX

        # Default to unknown
        return TypeComplexity.UNKNOWN

    @classmethod
    def _extract_base_type(cls, type_hint: Any, origin: Any, args: Tuple,
                           complexity: TypeComplexity) -> Type:
        """Extract the most fundamental base type from a complex type."""

        if complexity == TypeComplexity.PRIMITIVE:
            return type_hint if type_hint is not None else type(None)

        if complexity == TypeComplexity.BUILTIN_COLLECTION:
            if origin:
                return origin
            return type_hint

        if complexity == TypeComplexity.UNION:
            # For Union types, find the most common base type
            if args:
                # Filter out NoneType for Optional types
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return cls._extract_base_type(non_none_args[0],
                                                  get_origin(non_none_args[0]),
                                                  get_args(non_none_args[0]),
                                                  cls._determine_complexity(
                                                      non_none_args[0],
                                                      get_origin(
                                                          non_none_args[0]),
                                                      get_args(
                                                          non_none_args[0])))

                # For multiple non-None types, try to find common base
                return cls._find_common_base_type(non_none_args)
            return Any

        if complexity == TypeComplexity.CALLABLE:
            return Callable

        if complexity == TypeComplexity.TYPING_GENERIC:
            if origin:
                return origin
            # Try to extract from typing generics
            if hasattr(type_hint, '__origin__'):
                return type_hint.__origin__
            return Any

        if complexity == TypeComplexity.CLASS:
            return type_hint

        # For complex types, try to extract the origin
        if origin:
            return origin

        return Any

    @classmethod
    def _find_common_base_type(cls, types: List[Type]) -> Type:
        """Find the common base type for a list of types."""
        if not types:
            return Any

        if len(types) == 1:
            return types[0]

        # Check if all types are the same
        if all(t == types[0] for t in types):
            return types[0]

        # Check for common primitive patterns
        primitives = [t for t in types if t in cls.PRIMITIVE_TYPES]
        if primitives and len(primitives) == len(types):
            # All primitives - check for numeric types
            numeric_types = {int, float, complex}
            if all(t in numeric_types for t in primitives):
                return float  # Most general numeric type
            return object  # General object for mixed primitives

        # Try to find common MRO
        try:
            # Get method resolution orders
            mros = [inspect.getmro(t) if inspect.isclass(t) else (t,) for t in
                    types]

            # Find common classes in all MROs
            if mros:
                common_classes = set(mros[0])
                for mro in mros[1:]:
                    common_classes.intersection_update(mro)

                # Return the most specific common class (excluding object)
                common_list = list(common_classes)
                if object in common_list and len(common_list) > 1:
                    common_list.remove(object)

                if common_list:
                    # Sort by MRO position to get most specific
                    return min(common_list,
                               key=lambda cls: mros[0].index(cls) if cls in
                                                                     mros[
                                                                         0] else float(
                                   'inf'))

        except (TypeError, AttributeError):
            pass

        return object

    @classmethod
    def _is_optional_type(cls, type_hint: Any, origin: Any,
                          args: Tuple) -> bool:
        """Check if a type is Optional (Union with None)."""
        if origin is Union and args:
            return type(None) in args
        return False

    @classmethod
    def _format_type_string(cls, type_hint: Any) -> str:
        """Format a type hint as a readable string."""
        if hasattr(type_hint, '__name__'):
            return type_hint.__name__

        # Handle special typing constructs
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            if len(args) == 2 and type(None) in args:
                # Optional type
                non_none = next(arg for arg in args if arg is not type(None))
                return f"Optional[{cls._format_type_string(non_none)}]"
            else:
                # Union type
                arg_strs = [cls._format_type_string(arg) for arg in args]
                return f"Union[{', '.join(arg_strs)}]"

        if origin and args:
            origin_name = getattr(origin, '__name__', str(origin))
            arg_strs = [cls._format_type_string(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"

        return str(type_hint)


class SignatureMapper:
    """
    Main interface for mapping function and class signatures to parameter type information.
    """

    def __init__(self, include_defaults: bool = True,
                 include_metadata: bool = True):
        """
        Initialize the signature mapper.

        Args:
            include_defaults: Whether to include default values in the mapping
            include_metadata: Whether to include additional metadata
        """
        self.include_defaults = include_defaults
        self.include_metadata = include_metadata
        self.type_reducer = TypeReducer()

    def map_function_signature(self, func: Callable) -> Dict[str, TypeInfo]:
        """
        Map a function's signature to parameter type information.

        Args:
            func: Function to analyze

        Returns:
            Dictionary mapping parameter names to TypeInfo objects
        """
        try:
            # Get signature and type hints
            sig = inspect.signature(func)
            type_hints = get_type_hints(func) if hasattr(func,
                                                         '__annotations__') else {}

            param_map = {}

            for param_name, param in sig.parameters.items():
                # Get type hint from annotations or signature
                type_hint = type_hints.get(param_name, param.annotation)

                # Analyze the type
                type_info = self.type_reducer.analyze_type(type_hint,
                                                           param_name)

                # Add default value information
                if self.include_defaults and param.default is not inspect.Parameter.empty:
                    type_info.default_value = param.default

                # Add parameter metadata
                if self.include_metadata:
                    type_info.metadata.update({
                        'kind': param.kind.name,
                        'has_default': param.default is not inspect.Parameter.empty,
                        'is_variadic': param.kind in (
                            param.VAR_POSITIONAL, param.VAR_KEYWORD)
                    })

                param_map[param_name] = type_info

            return param_map

        except Exception as e:
            # Fallback for problematic functions
            return self._fallback_function_mapping(func, str(e))

    def map_class_signature(self, cls: Type, method_name: str = '__init__') -> \
            Dict[str, TypeInfo]:
        """
        Map a class constructor or method signature to parameter type information.

        Args:
            cls: Class to analyze
            method_name: Method to analyze (default: '__init__')

        Returns:
            Dictionary mapping parameter names to TypeInfo objects
        """
        try:
            # Get the method
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                param_map = self.map_function_signature(method)

                # Remove 'self' parameter if present
                if 'self' in param_map:
                    del param_map['self']

                return param_map
            else:
                return {}

        except Exception as e:
            return self._fallback_class_mapping(cls, method_name, str(e))

    def map_callable_signature(self, callable_obj: Any) -> Dict[str, TypeInfo]:
        """
        Map any callable object's signature to parameter type information.

        Args:
            callable_obj: Any callable object (function, method, class, etc.)

        Returns:
            Dictionary mapping parameter names to TypeInfo objects
        """
        if inspect.isclass(callable_obj):
            return self.map_class_signature(callable_obj)
        elif callable(callable_obj):
            return self.map_function_signature(callable_obj)
        else:
            raise TypeError(f"Object {callable_obj} is not callable")

    def get_parameter_types_only(self, callable_obj: Any) -> Dict[str, Type]:
        """
        Get a simplified mapping of parameter names to their base types only.

        Args:
            callable_obj: Any callable object

        Returns:
            Dictionary mapping parameter names to base types
        """
        type_map = self.map_callable_signature(callable_obj)
        return {name: info.base_type for name, info in type_map.items()}

    def get_parameter_complexity_analysis(self, callable_obj: Any) -> Dict[
        TypeComplexity, List[str]]:
        """
        Analyze parameter type complexity and group by complexity level.

        Args:
            callable_obj: Any callable object

        Returns:
            Dictionary mapping complexity levels to parameter names
        """
        type_map = self.map_callable_signature(callable_obj)
        complexity_map = defaultdict(list)

        for name, info in type_map.items():
            complexity_map[info.complexity].append(name)

        return dict(complexity_map)

    def _fallback_function_mapping(self, func: Callable, error: str) -> Dict[
        str, TypeInfo]:
        """Fallback mapping when signature analysis fails."""
        try:
            # Try basic signature without type hints
            sig = inspect.signature(func)
            param_map = {}

            for param_name, param in sig.parameters.items():
                type_info = TypeInfo(
                    name=param_name,
                    original_type=Any,
                    base_type=Any,
                    complexity=TypeComplexity.UNKNOWN,
                    annotation_string="Any",
                    metadata={'error': error, 'fallback': True}
                )

                if self.include_defaults and param.default is not inspect.Parameter.empty:
                    type_info.default_value = param.default

                param_map[param_name] = type_info

            return param_map

        except Exception:
            return {}

    def _fallback_class_mapping(self, cls: Type, method_name: str,
                                error: str) -> Dict[str, TypeInfo]:
        """Fallback mapping when class analysis fails."""
        return {
            'fallback': TypeInfo(
                name='fallback',
                original_type=Any,
                base_type=Any,
                complexity=TypeComplexity.UNKNOWN,
                annotation_string="Any",
                metadata={'error': error, 'class': cls.__name__,
                          'method': method_name}
            )
        }


# Utility functions for common use cases

def get_parameter_types(callable_obj: Any, simple: bool = False) -> Dict[
    str, Any]:
    """
    Quick utility to get parameter types from any callable.

    Args:
        callable_obj: Any callable object
        simple: If True, return only base types; if False, return TypeInfo objects

    Returns:
        Dictionary mapping parameter names to types or TypeInfo objects
    """
    mapper = SignatureMapper()

    if simple:
        return mapper.get_parameter_types_only(callable_obj)
    else:
        return mapper.map_callable_signature(callable_obj)


def analyze_type_complexity(callable_obj: Any) -> Dict[str, str]:
    """
    Analyze the complexity of all parameters in a callable.

    Args:
        callable_obj: Any callable object

    Returns:
        Dictionary mapping parameter names to complexity descriptions
    """
    mapper = SignatureMapper()
    type_map = mapper.map_callable_signature(callable_obj)

    return {
        name: f"{info.complexity.value} ({info.annotation_string})"
        for name, info in type_map.items()
    }


# Example usage and demonstration

def example_complex_function(
        name: str,
        age: int = 25,
        hobbies: List[str] = None,
        metadata: Dict[str, Any] = None,
        callback: Optional[Callable[[str], bool]] = None,
        *args,
        score: Union[int, float] = 0.0,
        **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """Example function with complex type annotations."""
    return name, {"age": age, "hobbies": hobbies or []}

# class ExampleClass:
#     """Example class with complex constructor."""
#
#     def __init__(
#             self,
#             name: str,
#             values: List[Union[int, float]],
#             mapping: Dict[str, Optional[Callable]],
#             config: Optional['ExampleClass'] = None
#     ):
#         self.name = name
#         self.values = values
#         self.mapping = mapping
#         self.config = config


# Demonstration
# if __name__ == "__main__":
#     print("=== Signature Type Mapping Demo ===\n")
#
#     mapper = SignatureMapper()
#
#     # Test function mapping
#     print("1. Function Signature Analysis:")
#     func_types = mapper.map_function_signature(example_complex_function)
#     for name, type_info in func_types.items():
#         print(
#             f"  {name}: {type_info.annotation_string} (base: {type_info.base_type.__name__}, complexity: {type_info.complexity.value})")
#
#     print("\n2. Class Constructor Analysis:")
#     class_types = mapper.map_class_signature(ChartSettings)
#     for name, type_info in class_types.items():
#         print(
#             f"  {name}: {type_info.annotation_string} (base: {type_info.base_type.__name__}, complexity: {type_info.complexity.value})")
#
#     print("\n3. Complexity Analysis:")
#     complexity_analysis = mapper.get_parameter_complexity_analysis(
#         example_complex_function)
#     for complexity, params in complexity_analysis.items():
#         print(f"  {complexity.value}: {params}")
#
#     print("\n4. Simple Type Mapping:")
#     simple_types = get_parameter_types(example_complex_function, simple=True)
#     for name, type_obj in simple_types.items():
#         print(f"  {name}: {type_obj.__name__}")
