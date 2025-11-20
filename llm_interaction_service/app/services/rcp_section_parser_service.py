import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RCPSection:
    """Represents a section in an RCP document"""
    number: str  # e.g., "4.1", "4.2"
    title: str   # e.g., "Indicații terapeutice"
    content: str
    start_pos: int
    end_pos: int

class RCPSectionParserService:
    """
    Service for parsing RCP documents into sections.
    RCP documents follow a standard structure with numbered sections.
    """
    
    # Standard RCP section patterns
    SECTION_PATTERN = re.compile(
        r'^(\d+\.(?:\d+)?)\s+([A-ZĂÂÎȘȚ][A-ZĂÂÎȘȚ\s,\-\/]+(?:\([^)]*\))?)\s*$',
        re.MULTILINE
    )
    
    # Alternative pattern for sections without titles
    SECTION_NUMBER_PATTERN = re.compile(
        r'^(\d+\.(?:\d+)?)\s*$',
        re.MULTILINE
    )
    
    def __init__(self):
        """Initialize RCP section parser."""
        logger.info("RCP Section Parser initialized")
    
    def parse_sections(self, text: str) -> List[RCPSection]:
        """
        Parse RCP text into structured sections.
        
        Args:
            text: Full RCP document text
            
        Returns:
            List of RCPSection objects
        """
        sections = []
        
        # Find all section headers
        matches = list(self.SECTION_PATTERN.finditer(text))
        
        if not matches:
            logger.warning("No section headers found in document")
            # Return entire document as single section
            return [RCPSection(
                number="0",
                title="Document complet",
                content=text,
                start_pos=0,
                end_pos=len(text)
            )]
        
        # Extract sections
        for i, match in enumerate(matches):
            section_number = match.group(1)
            section_title = match.group(2).strip()
            start_pos = match.start()
            
            # Determine end position (start of next section or end of document)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # Extract content (excluding the header line)
            content_start = match.end()
            content = text[content_start:end_pos].strip()
            
            sections.append(RCPSection(
                number=section_number,
                title=section_title,
                content=content,
                start_pos=start_pos,
                end_pos=end_pos
            ))
        
        logger.info(f"Parsed {len(sections)} sections from document")
        return sections
    
    def get_section_by_number(self, sections: List[RCPSection], number: str) -> Optional[RCPSection]:
        """
        Get a specific section by its number.
        
        Args:
            sections: List of parsed sections
            number: Section number (e.g., "4.1")
            
        Returns:
            RCPSection if found, None otherwise
        """
        for section in sections:
            if section.number == number:
                return section
        return None
    
    def get_sections_by_prefix(self, sections: List[RCPSection], prefix: str) -> List[RCPSection]:
        """
        Get all sections matching a prefix.
        
        Args:
            sections: List of parsed sections
            prefix: Section number prefix (e.g., "4" returns 4.1, 4.2, etc.)
            
        Returns:
            List of matching RCPSection objects
        """
        return [s for s in sections if s.number.startswith(prefix)]
    
    def create_chunks_from_sections(
        self,
        sections: List[RCPSection],
        max_chunk_size: int = 512,
        overlap: int = 100,
        preserve_sections: bool = True
    ) -> List[Dict[str, any]]:
        """
        Create chunks from sections with metadata.
        
        Args:
            sections: List of RCPSection objects
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            preserve_sections: If True, don't split across section boundaries
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(
                section,
                max_chunk_size,
                overlap,
                preserve_sections
            )
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _chunk_section(
        self,
        section: RCPSection,
        max_chunk_size: int,
        overlap: int,
        preserve_section: bool
    ) -> List[Dict[str, any]]:
        """
        Create chunks from a single section.
        
        Args:
            section: RCPSection to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap
            preserve_section: Don't split if possible
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        content = section.content
        
        # If section is small enough, keep as single chunk
        if len(content) <= max_chunk_size or preserve_section:
            chunks.append({
                'text': content,
                'metadata': {
                    'section_number': section.number,
                    'section_title': section.title,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            })
            return chunks
        
        # Split into multiple chunks with overlap
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for period, question mark, or exclamation within last 100 chars
                last_section = content[max(end - 100, start):end]
                sentence_end = max(
                    last_section.rfind('.'),
                    last_section.rfind('!'),
                    last_section.rfind('?')
                )
                if sentence_end != -1:
                    end = max(end - 100, start) + sentence_end + 1
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'section_number': section.number,
                        'section_title': section.title,
                        'chunk_index': chunk_index,
                        'total_chunks': None  # Will be filled after all chunks created
                    }
                })
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - overlap
        
        # Update total_chunks for all chunks in this section
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def extract_key_sections(self, sections: List[RCPSection]) -> Dict[str, RCPSection]:
        """
        Extract commonly queried sections from RCP.
        
        Args:
            sections: List of all sections
            
        Returns:
            Dictionary mapping section type to RCPSection
        """
        key_sections = {}
        
        # Common RCP sections
        section_map = {
            '1': 'denumire_comerciala',
            '2': 'compozitie',
            '3': 'forma_farmaceutica',
            '4.1': 'indicatii_terapeutice',
            '4.2': 'doze_mod_administrare',
            '4.3': 'contraindicatii',
            '4.4': 'atentionari_precautii',
            '4.5': 'interactiuni',
            '4.6': 'fertilitate_sarcina_alaptare',
            '4.7': 'efecte_capacitate_condus',
            '4.8': 'reactii_adverse',
            '4.9': 'supradozaj',
            '5': 'proprietati_farmacologice',
            '6': 'proprietati_farmaceutice'
        }
        
        for section in sections:
            section_key = section_map.get(section.number)
            if section_key:
                key_sections[section_key] = section
        
        return key_sections
    
    def get_section_statistics(self, sections: List[RCPSection]) -> Dict[str, any]:
        """
        Get statistics about parsed sections.
        
        Args:
            sections: List of sections
            
        Returns:
            Dictionary with statistics
        """
        if not sections:
            return {}
        
        total_chars = sum(len(s.content) for s in sections)
        
        return {
            'total_sections': len(sections),
            'total_characters': total_chars,
            'avg_section_length': total_chars / len(sections) if sections else 0,
            'min_section_length': min(len(s.content) for s in sections),
            'max_section_length': max(len(s.content) for s in sections),
            'section_numbers': [s.number for s in sections],
            'section_titles': [s.title for s in sections]
        }
