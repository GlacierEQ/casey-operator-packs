#!/usr/bin/env python3
"""
Comet Browser Assistant - MCP Browser Controller
Legal analysis, citation extraction, and evidence collection for web content
Designed for Casey's AI Legal Empire with chain-of-custody compliance
"""

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse
import logging

# Web scraping and parsing imports
try:
    import requests
    from bs4 import BeautifulSoup
    import lxml
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("Web scraping not available - install: pip install requests beautifulsoup4 lxml")

# Legal text processing
try:
    import spacy
    import re
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("NLP not available - install: pip install spacy")

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Web content analysis types"""
    OPINION = "opinion"           # Court opinions and decisions
    STATUTE = "statute"           # Legal statutes and codes
    FORM = "form"                # Legal forms and templates  
    EVIDENCE = "evidence"         # Evidence materials
    NEWS = "news"                # Legal news and articles
    ACADEMIC = "academic"        # Law review articles
    GENERAL = "general"          # General legal content

class AdmissibilityScore(Enum):
    """Evidence admissibility scoring"""
    HIGHLY_ADMISSIBLE = "highly_admissible"     # Strong evidentiary value
    LIKELY_ADMISSIBLE = "likely_admissible"     # Probable admission with foundation
    QUESTIONABLE = "questionable"               # Requires careful analysis
    LIKELY_INADMISSIBLE = "likely_inadmissible" # Probable exclusion
    INADMISSIBLE = "inadmissible"               # Clear exclusion under rules

@dataclass
class WebPageAnalysis:
    """Complete web page legal analysis"""
    url: str
    title: str
    content_type: AnalysisType
    legal_analysis: str
    admissibility_score: AdmissibilityScore
    admissibility_reasoning: str
    citations_found: List[str]
    key_passages: List[str]
    contradictions_flagged: List[str]
    related_authorities: List[str]
    content_hash: str
    analysis_timestamp: float = field(default_factory=time.time)
    provenance_signature: str = field(default_factory=lambda: hashlib.sha256(f"analysis_{time.time()}".encode()).hexdigest())

@dataclass
class EvidenceItem:
    """Evidence item with chain of custody"""
    evidence_url: str
    evidence_type: str
    content_snapshot: str
    content_hash: str
    custody_metadata: Dict[str, Any]
    collection_timestamp: float = field(default_factory=time.time)
    chain_of_custody_id: str = field(default_factory=lambda: hashlib.sha256(f"evidence_{time.time()}".encode()).hexdigest()[:16])
    admissibility_assessment: Optional[AdmissibilityScore] = None
    custody_signature: str = field(default_factory=str)
    
    def __post_init__(self):
        """Generate custody signature after initialization"""
        if not self.custody_signature:
            custody_data = {
                "url": self.evidence_url,
                "hash": self.content_hash,
                "timestamp": self.collection_timestamp,
                "metadata": self.custody_metadata
            }
            self.custody_signature = hashlib.sha256(
                json.dumps(custody_data, sort_keys=True).encode()
            ).hexdigest()

@dataclass
class ResearchSession:
    """Browser research session tracking"""
    session_id: str
    case_id: str
    query: str
    suggested_urls: List[str]
    legal_authorities: List[str] 
    research_strategy: str
    pages_analyzed: List[WebPageAnalysis] = field(default_factory=list)
    evidence_collected: List[EvidenceItem] = field(default_factory=list)
    session_start: float = field(default_factory=time.time)
    total_findings: int = 0

class LegalContentDetector:
    """Detect and classify legal content on web pages"""
    
    COURT_INDICATORS = [
        r'\b(court|judge|justice|magistrate)\b',
        r'\b(opinion|decision|ruling|order|judgment)\b',
        r'\b(plaintiff|defendant|appellant|appellee)\b',
        r'\b(case\s+no\.?|docket|civil\s+action)\b'
    ]
    
    STATUTE_INDICATORS = [
        r'\b(statute|code|section|subsection)\b',
        r'\b(\d+\s*U\.?S\.?C\.?|USC)\b',
        r'\b(HRS|Hawaii\s+Revised\s+Statutes)\b',
        r'\b(\u00a7|section)\s*\d+'
    ]
    
    EVIDENCE_INDICATORS = [
        r'\b(evidence|exhibit|testimony|affidavit)\b',
        r'\b(document|record|transcript|deposition)\b',
        r'\b(authenticated|certified|notarized)\b'
    ]
    
    CASE_SPECIFIC_TERMS = [
        r'\b(custody|parental\s+rights|best\s+interest)\b',
        r'\b(family\s+court|domestic\s+relations)\b',
        r'\b(child\s+welfare|protective\s+services)\b',
        r'\b1FDV-23-0001009\b'
    ]
    
    @classmethod
    def detect_content_type(cls, content: str, url: str) -> AnalysisType:
        """Detect the type of legal content"""
        content_lower = content.lower()
        
        # Court opinion detection
        court_score = sum(1 for pattern in cls.COURT_INDICATORS 
                         if re.search(pattern, content_lower, re.IGNORECASE))
        
        # Statute detection  
        statute_score = sum(1 for pattern in cls.STATUTE_INDICATORS
                           if re.search(pattern, content_lower, re.IGNORECASE))
        
        # Evidence detection
        evidence_score = sum(1 for pattern in cls.EVIDENCE_INDICATORS
                            if re.search(pattern, content_lower, re.IGNORECASE))
        
        # URL-based hints
        if 'courtlistener' in url.lower() or 'courts' in url.lower():
            return AnalysisType.OPINION
        elif 'statute' in url.lower() or 'code' in url.lower():
            return AnalysisType.STATUTE
        elif 'form' in url.lower():
            return AnalysisType.FORM
            
        # Score-based classification
        if court_score >= 3:
            return AnalysisType.OPINION
        elif statute_score >= 2:
            return AnalysisType.STATUTE
        elif evidence_score >= 2:
            return AnalysisType.EVIDENCE
        else:
            return AnalysisType.GENERAL
    
    @classmethod
    def extract_citations(cls, content: str) -> List[str]:
        """Extract legal citations from content"""
        citations = []
        
        # Federal citations
        federal_patterns = [
            r'\d+\s+U\.?S\.?\s+\d+',  # U.S. citations
            r'\d+\s+F\.?\d*d?\s+\d+',  # Federal Reporter
            r'\d+\s+S\.?\s*Ct\.?\s+\d+',  # Supreme Court Reporter
        ]
        
        # Hawaii citations
        hawaii_patterns = [
            r'\d+\s+Haw\.?\s+\d+',  # Hawaii Reports
            r'HRS\s*\u00a7?\s*\d+[\-\.]?\d*',  # Hawaii Revised Statutes
            r'HFCR\s+Rule\s+\d+',  # Hawaii Family Court Rules
            r'HRE\s+Rule\s+\d+',  # Hawaii Rules of Evidence
        ]
        
        all_patterns = federal_patterns + hawaii_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend(matches)
            
        return list(set(citations))  # Remove duplicates
    
    @classmethod
    def assess_admissibility(cls, content: str, content_type: AnalysisType, url: str) -> tuple[AdmissibilityScore, str]:
        """Assess evidence admissibility based on content analysis"""
        content_lower = content.lower()
        
        # High admissibility indicators
        if content_type == AnalysisType.OPINION:
            if any(term in content_lower for term in ['published', 'precedential', 'citable']):
                return AdmissibilityScore.HIGHLY_ADMISSIBLE, "Published court opinion with precedential value"
        
        # Government/official sources
        if any(domain in url.lower() for domain in ['.gov', 'courts.', 'legislature.']):
            return AdmissibilityScore.LIKELY_ADMISSIBLE, "Official government source"
            
        # Academic sources
        if any(domain in url.lower() for domain in ['.edu', 'lawreview', 'journal']):
            return AdmissibilityScore.LIKELY_ADMISSIBLE, "Academic legal source"
            
        # News and commentary
        if any(term in url.lower() for term in ['news', 'blog', 'opinion', 'editorial']):
            return AdmissibilityScore.QUESTIONABLE, "News/commentary source - may need authentication"
            
        # Social media and informal sources
        if any(term in url.lower() for term in ['social', 'forum', 'reddit', 'facebook']):
            return AdmissibilityScore.LIKELY_INADMISSIBLE, "Informal source lacking authentication"
            
        # Default assessment
        return AdmissibilityScore.QUESTIONABLE, "Requires further analysis for admissibility determination"

class CourtListenerIntegration:
    """Integration with CourtListener API for case law research"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.courtlistener.com/api/rest/v3/"
        
    async def search_opinions(self, query: str, jurisdiction: str = None) -> List[Dict[str, Any]]:
        """Search court opinions via CourtListener API"""
        try:
            params = {
                'q': query,
                'format': 'json',
                'order_by': '-date_filed'
            }
            
            if jurisdiction:
                params['court__jurisdiction'] = jurisdiction
                
            # TODO: Implement actual API call
            # For now, return mock data
            return [{
                'case_name': 'Sample v. Case',
                'court': 'Hawaii Supreme Court',
                'date_filed': '2024-01-15',
                'citation': '2024 Haw. 123',
                'url': 'https://courtlistener.com/opinion/123456/',
                'snippet': 'Legal precedent relevant to custody matters...'
            }]
            
        except Exception as e:
            logger.error(f"CourtListener search failed: {e}")
            return []
    
    async def get_related_cases(self, case_citation: str) -> List[Dict[str, Any]]:
        """Get cases related to a specific citation"""
        try:
            # TODO: Implement related case lookup
            return [{
                'case_name': 'Related v. Case',
                'citation': '2023 Haw. 456',
                'relationship': 'citing'
            }]
            
        except Exception as e:
            logger.error(f"Related case lookup failed: {e}")
            return []

class MCPBrowserController:
    """Main browser controller with MCP integration"""
    
    def __init__(self, courtlistener_api_key: str = None):
        self.content_detector = LegalContentDetector()
        self.courtlistener = CourtListenerIntegration(courtlistener_api_key) if courtlistener_api_key else None
        
    async def analyze_web_page(self, url: str, content_html: str, analysis_type: str = None) -> WebPageAnalysis:
        """Analyze web page for legal content"""
        try:
            # Parse HTML content
            if WEB_SCRAPING_AVAILABLE:
                soup = BeautifulSoup(content_html, 'html.parser')
                title = soup.title.string if soup.title else "Untitled"
                text_content = soup.get_text()
            else:
                title = "Content Analysis"
                text_content = content_html  # Fallback to raw HTML
            
            # Detect content type
            if analysis_type:
                content_type = AnalysisType(analysis_type)
            else:
                content_type = self.content_detector.detect_content_type(text_content, url)
            
            # Extract citations
            citations_found = self.content_detector.extract_citations(text_content)
            
            # Assess admissibility
            admissibility_score, admissibility_reasoning = self.content_detector.assess_admissibility(
                text_content, content_type, url
            )
            
            # Generate key passages (simplified)
            key_passages = self._extract_key_passages(text_content, content_type)
            
            # Flag contradictions (placeholder)
            contradictions_flagged = self._detect_contradictions(text_content)
            
            # Find related authorities
            related_authorities = await self._find_related_authorities(text_content, citations_found)
            
            # Generate content hash for integrity
            content_hash = hashlib.sha256(content_html.encode()).hexdigest()
            
            # Create legal analysis
            legal_analysis = self._generate_legal_analysis(
                content_type, admissibility_score, citations_found, key_passages
            )
            
            return WebPageAnalysis(
                url=url,
                title=title,
                content_type=content_type,
                legal_analysis=legal_analysis,
                admissibility_score=admissibility_score,
                admissibility_reasoning=admissibility_reasoning,
                citations_found=citations_found,
                key_passages=key_passages,
                contradictions_flagged=contradictions_flagged,
                related_authorities=related_authorities,
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Web page analysis failed: {e}")
            raise
    
    async def conduct_research(self, query: str, case_id: str, context: Dict[str, Any]) -> ResearchSession:
        """Conduct legal research with suggested authorities"""
        try:
            session_id = f"research_{int(time.time())}"
            
            # Generate research strategy
            research_strategy = self._generate_research_strategy(query, case_id)
            
            # Get suggested URLs
            suggested_urls = await self._generate_suggested_urls(query, case_id)
            
            # Find legal authorities via CourtListener
            legal_authorities = []
            if self.courtlistener:
                opinions = await self.courtlistener.search_opinions(query, 'hawaii')
                legal_authorities = [f"{op['case_name']} ({op['citation']})" for op in opinions]
            
            return ResearchSession(
                session_id=session_id,
                case_id=case_id,
                query=query,
                suggested_urls=suggested_urls,
                legal_authorities=legal_authorities,
                research_strategy=research_strategy
            )
            
        except Exception as e:
            logger.error(f"Research session failed: {e}")
            raise
    
    async def collect_evidence(self, evidence_url: str, evidence_type: str, 
                              chain_metadata: Dict[str, Any]) -> EvidenceItem:
        """Collect evidence with chain of custody"""
        try:
            # Fetch content in E2B sandbox (simulated)
            if WEB_SCRAPING_AVAILABLE:
                response = requests.get(evidence_url, timeout=30)
                content_snapshot = response.text
            else:
                content_snapshot = f"Content from {evidence_url}"  # Placeholder
            
            # Generate content hash
            content_hash = hashlib.sha256(content_snapshot.encode()).hexdigest()
            
            # Assess admissibility
            _, content_type_detected = self.content_detector.detect_content_type(content_snapshot, evidence_url), None
            admissibility_score, _ = self.content_detector.assess_admissibility(
                content_snapshot, content_type_detected or AnalysisType.EVIDENCE, evidence_url
            )
            
            return EvidenceItem(
                evidence_url=evidence_url,
                evidence_type=evidence_type,
                content_snapshot=content_snapshot,
                content_hash=content_hash,
                custody_metadata=chain_metadata,
                admissibility_assessment=admissibility_score
            )
            
        except Exception as e:
            logger.error(f"Evidence collection failed: {e}")
            raise
    
    def _extract_key_passages(self, content: str, content_type: AnalysisType) -> List[str]:
        """Extract key legal passages from content"""
        # Simplified key passage extraction
        sentences = content.split('. ')
        
        # Look for important legal terms
        key_terms = ['custody', 'due process', 'best interest', 'parental rights', 'evidence']
        
        key_passages = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in key_terms):
                if len(sentence) > 50 and len(sentence) < 500:  # Reasonable length
                    key_passages.append(sentence.strip())
                    
        return key_passages[:5]  # Return top 5 passages
    
    def _detect_contradictions(self, content: str) -> List[str]:
        """Detect potential contradictions in content"""
        # Placeholder contradiction detection
        contradictions = []
        
        # Look for conflicting statements
        if 'but' in content.lower() or 'however' in content.lower():
            contradictions.append("Potential conflicting statements detected")
            
        return contradictions
    
    async def _find_related_authorities(self, content: str, citations: List[str]) -> List[str]:
        """Find related legal authorities"""
        related = []
        
        # Use CourtListener for related cases
        if self.courtlistener and citations:
            for citation in citations[:3]:  # Limit to first 3 citations
                related_cases = await self.courtlistener.get_related_cases(citation)
                related.extend([case['case_name'] for case in related_cases])
                
        return related[:10]  # Limit to 10 related authorities
    
    def _generate_legal_analysis(self, content_type: AnalysisType, 
                                admissibility: AdmissibilityScore,
                                citations: List[str], key_passages: List[str]) -> str:
        """Generate legal analysis summary"""
        analysis_parts = []
        
        analysis_parts.append(f"Content Type: {content_type.value.title()}")
        analysis_parts.append(f"Admissibility Assessment: {admissibility.value.replace('_', ' ').title()}")
        
        if citations:
            analysis_parts.append(f"Legal Citations Found: {len(citations)} authorities referenced")
            
        if key_passages:
            analysis_parts.append(f"Key Legal Concepts: {len(key_passages)} relevant passages identified")
            
        # Case-specific analysis
        if any('custody' in passage.lower() for passage in key_passages):
            analysis_parts.append("Custody-related content relevant to Case 1FDV-23-0001009")
            
        return ". ".join(analysis_parts) + "."
    
    def _generate_research_strategy(self, query: str, case_id: str) -> str:
        """Generate research strategy based on query"""
        if 'custody' in query.lower():
            return "Focus on Hawaii family law precedents, federal constitutional protections, and best interest standards"
        elif 'evidence' in query.lower():
            return "Research Hawaii Rules of Evidence, authentication requirements, and admissibility standards"
        elif 'due process' in query.lower():
            return "Examine 14th Amendment jurisprudence, procedural due process requirements, and family court procedures"
        else:
            return "Comprehensive legal research targeting relevant authorities and precedents"
    
    async def _generate_suggested_urls(self, query: str, case_id: str) -> List[str]:
        """Generate suggested research URLs"""
        # Base legal research URLs
        urls = [
            "https://www.courtlistener.com/",
            "https://www.capitol.hawaii.gov/hrscurrent/",
            "https://www.courts.state.hi.us/"
        ]
        
        # Query-specific URLs
        if 'custody' in query.lower():
            urls.extend([
                "https://www.capitol.hawaii.gov/hrscurrent/Vol12_Ch0501-0588/HRS0571/",
                "https://www.courts.state.hi.us/legal_references/rules/hfcr"
            ])
            
        return urls

# API endpoint functions
async def analyze_endpoint(url: str, content_html: str, analysis_type: str = None) -> Dict[str, Any]:
    """Browser analysis API endpoint"""
    controller = MCPBrowserController()
    analysis = await controller.analyze_web_page(url, content_html, analysis_type)
    
    return {
        "legal_analysis": analysis.legal_analysis,
        "admissibility_score": analysis.admissibility_score.value,
        "admissibility_reasoning": analysis.admissibility_reasoning,
        "citations_found": analysis.citations_found,
        "key_passages": analysis.key_passages,
        "content_hash": analysis.content_hash,
        "provenance_signature": analysis.provenance_signature
    }

async def research_endpoint(query: str, case_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Browser research API endpoint"""
    controller = MCPBrowserController()
    session = await controller.conduct_research(query, case_id, context)
    
    return {
        "session_id": session.session_id,
        "suggested_urls": session.suggested_urls,
        "legal_authorities": session.legal_authorities,
        "research_strategy": session.research_strategy
    }

async def evidence_endpoint(evidence_url: str, evidence_type: str, 
                           chain_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Browser evidence collection API endpoint"""
    controller = MCPBrowserController()
    evidence = await controller.collect_evidence(evidence_url, evidence_type, chain_metadata)
    
    return {
        "custody_receipt": {
            "chain_of_custody_id": evidence.chain_of_custody_id,
            "collection_timestamp": evidence.collection_timestamp,
            "evidence_url": evidence.evidence_url
        },
        "hash_signature": evidence.custody_signature,
        "content_hash": evidence.content_hash,
        "admissibility_assessment": evidence.admissibility_assessment.value if evidence.admissibility_assessment else None
    }

# Export key classes and functions
__all__ = [
    'MCPBrowserController',
    'AnalysisType',
    'AdmissibilityScore', 
    'WebPageAnalysis',
    'EvidenceItem',
    'analyze_endpoint',
    'research_endpoint',
    'evidence_endpoint'
]

# Example usage for Case 1FDV-23-0001009
if __name__ == "__main__":
    async def main():
        controller = MCPBrowserController()
        
        # Analyze a legal document
        sample_html = "<html><body>This court finds that custody arrangements must prioritize the best interests of the child per HRS §571-46.</body></html>"
        analysis = await controller.analyze_web_page(
            "https://example.com/court-opinion",
            sample_html
        )
        
        print(f"Content Type: {analysis.content_type.value}")
        print(f"Admissibility: {analysis.admissibility_score.value}")
        print(f"Citations: {analysis.citations_found}")
        print(f"Analysis: {analysis.legal_analysis}")
        
    asyncio.run(main())