import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Set, Callable, Optional, Any, List


class FileExtensionCategory(Enum):
    """File type categories"""
    IMAGE = "image"
    DOCUMENT = "document"
    CODE = "code"
    SPREADSHEET = "spreadsheet"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Immutable file information container with optimized memory usage"""
    path: Path
    name: str
    size: int
    mime_type: str
    category: FileExtensionCategory
    line_count: Optional[int] = None
    preview_available: bool = False


class FileTypeHelper:
    """Helper class for file type detection and categorization"""

    # Use frozenset for immutable, optimized lookups
    CATEGORY_EXTENSIONS: Dict[FileExtensionCategory, frozenset] = {
        FileExtensionCategory.IMAGE: frozenset({
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'
        }),
        FileExtensionCategory.DOCUMENT: frozenset({
            '.txt', '.md', '.doc', '.docx', '.odt', '.rtf', '.pptx', '.ppt'
        }),
        FileExtensionCategory.CODE: frozenset({
            '.py', '.ipynb', '.js', '.java', '.cpp', '.c', '.h', '.css',
            '.html', '.htm', '.xml', '.json', '.yaml', '.yml', '.sh', '.rs',
            '.go'
        }),
        FileExtensionCategory.SPREADSHEET: frozenset({
            '.csv', '.xls', '.xlsx', '.ods'
        }),
        FileExtensionCategory.PDF: frozenset({'.pdf'}),
        FileExtensionCategory.VIDEO: frozenset({
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'
        }),
        FileExtensionCategory.AUDIO: frozenset({
            '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a'
        }),
        FileExtensionCategory.ARCHIVE: frozenset({
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2'
        }),
    }

    TEXT_EXTENSIONS: frozenset = frozenset({
        '.txt', '.md', '.py', '.js', '.css', '.html', '.xml',
        '.json', '.yaml', '.yml', '.csv', '.log', '.ini', '.cfg'
    })

    # Create reverse lookup for O(1) category access
    _EXT_TO_CATEGORY: Dict[str, FileExtensionCategory] = {
        ext: category
        for category, extensions in CATEGORY_EXTENSIONS.items()
        for ext in extensions
    }

    PREVIEWABLE_IMAGES: frozenset = frozenset({
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'
    })

    @classmethod
    def get_category(cls, file_path: Path) -> FileExtensionCategory:
        """Determine file category from extension (cached)"""
        ext = file_path.suffix.lower()
        return cls._EXT_TO_CATEGORY.get(ext, FileExtensionCategory.UNKNOWN)

    @classmethod
    def can_count_lines(cls, file_path: Path) -> bool:
        """Check if file is text-based for line counting"""
        return file_path.suffix.lower() in cls.TEXT_EXTENSIONS

    @classmethod
    def get_mime_type(cls, file_path: Path) -> str:
        """Get MIME type of file (cached)"""
        # mimetypes.guess_type returns (type, encoding)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'

    @classmethod
    def can_preview_image(cls, file_path: Path) -> bool:
        """Check if image preview is possible"""
        return file_path.suffix.lower() in cls.PREVIEWABLE_IMAGES


class FileExtensionManager:
    """
    Centralized manager for file extension handling throughout the codebase.
    """

    __slots__ = ()

    # Registry for file extensions and their handlers
    _extension_registry: Dict[str, Dict[str, Any]] = {}
    _category_registry: Dict[FileExtensionCategory, Set[str]] = {}
    _handler_registry: Dict[str, Callable] = {}

    @classmethod
    def register_extension(
            cls,
            extension: str,
            category: FileExtensionCategory,
            handler: Optional[Callable] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a file extension with its category and optional handler.

        Args:
            extension: File extension with dot (e.g., '.json')
            category: Category from FileExtensionCategory enum
            handler: Optional handler function for this extension
            metadata: Optional metadata about the extension
        """
        # Normalize extension
        ext = extension if extension.startswith('.') else f'.{extension}'
        ext = ext.lower()

        cls._extension_registry[ext] = {
            'category': category,
            'handler': handler,
            'metadata': metadata or {}
        }

        # Use setdefault to avoid redundant checks
        cls._category_registry.setdefault(category, set()).add(ext)

        if handler:
            cls._handler_registry[ext] = handler

    @classmethod
    def get_category(cls, extension: str) -> FileExtensionCategory:
        """Get the category for a file extension"""
        ext = extension if extension.startswith('.') else f'.{extension}'
        entry = cls._extension_registry.get(ext.lower())
        return entry['category'] if entry else FileExtensionCategory.UNKNOWN

    @classmethod
    def get_handler(cls, extension: str) -> Optional[Callable]:
        """Get the handler for a file extension"""
        ext = extension if extension.startswith('.') else f'.{extension}'
        return cls._handler_registry.get(ext.lower())

    @classmethod
    def get_extensions_by_category(
            cls,
            category: FileExtensionCategory
    ) -> frozenset:
        """Get all extensions in a category (returns immutable set)"""
        return frozenset(cls._category_registry.get(category, set()))

    @classmethod
    def is_extension_registered(cls, extension: str) -> bool:
        """Check if an extension is registered"""
        ext = extension if extension.startswith('.') else f'.{extension}'
        return ext.lower() in cls._extension_registry

    @classmethod
    def get_metadata(cls, extension: str) -> Dict[str, Any]:
        """Get metadata for an extension"""
        ext = extension if extension.startswith('.') else f'.{extension}'
        entry = cls._extension_registry.get(ext.lower())
        return entry.get('metadata', {}) if entry else {}

    @classmethod
    def get_all_extensions(cls) -> List[str]:
        """Get all registered extensions"""
        return list(cls._extension_registry.keys())

    @classmethod
    def unregister_extension(cls, extension: str) -> bool:
        """
        Unregister an extension from the manager.

        Returns:
            bool: True if extension was unregistered, False if not found
        """
        ext = extension if extension.startswith('.') else f'.{extension}'
        ext = ext.lower()

        if ext not in cls._extension_registry:
            return False

        entry = cls._extension_registry.pop(ext)
        category = entry['category']

        if category in cls._category_registry:
            cls._category_registry[category].discard(ext)
            if not cls._category_registry[category]:
                del cls._category_registry[category]

        cls._handler_registry.pop(ext, None)
        return True

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered extensions"""
        cls._extension_registry.clear()
        cls._category_registry.clear()
        cls._handler_registry.clear()


def initialize_default_extensions() -> None:
    """Initialize the manager with common file extensions"""
    for category, extensions in FileTypeHelper.CATEGORY_EXTENSIONS.items():
        for extension in extensions:
            mime_type = mimetypes.types_map.get(extension,
                                                'application/octet-stream')

            is_text = (
                    mime_type.startswith('text/') or
                    extension in FileTypeHelper.TEXT_EXTENSIONS or
                    mime_type in {
                        'application/json',
                        'application/xml',
                        'application/x-yaml',
                        'application/x-sh'
                    }
            )

            FileExtensionManager.register_extension(
                extension,
                category,
                metadata={
                    'mime_type': mime_type,
                    'is_text': is_text
                }
            )


# Initialize the default extensions
initialize_default_extensions()