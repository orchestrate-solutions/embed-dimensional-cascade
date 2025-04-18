#!/usr/bin/env python
# Generate enhanced sample data with longer texts for dimensional cascade testing

import json
import argparse
import random
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Categories for generating domain-specific content
CATEGORIES = {
    "technology": [
        "artificial intelligence", "machine learning", "neural networks", "deep learning", 
        "natural language processing", "computer vision", "robotics", "automation",
        "cloud computing", "edge computing", "Internet of Things", "blockchain",
        "cybersecurity", "data science", "big data", "analytics", "algorithm",
        "vector embeddings", "dimensionality reduction", "search technology"
    ],
    "science": [
        "quantum physics", "astrophysics", "molecular biology", "genetics",
        "climate science", "neuroscience", "materials science", "computational chemistry",
        "particle physics", "nuclear fusion", "renewable energy", "sustainability",
        "ecosystem", "conservation", "genomics", "proteomics", "CRISPR",
        "nanotechnology", "biotechnology", "scientific method"
    ],
    "business": [
        "entrepreneurship", "startup", "venture capital", "investment strategy",
        "digital transformation", "business intelligence", "market analysis",
        "strategic planning", "supply chain management", "customer relationship",
        "financial analysis", "risk management", "corporate governance", 
        "innovation management", "digital marketing", "e-commerce",
        "product development", "business model", "competitive analysis", "KPIs"
    ],
    "healthcare": [
        "precision medicine", "telemedicine", "electronic health records", "medical imaging",
        "healthcare analytics", "public health", "epidemiology", "clinical trials",
        "drug discovery", "pharmaceutical research", "biomedical engineering",
        "healthcare policy", "patient care", "medical diagnostics", "treatment protocols",
        "disease prevention", "wellness programs", "mental health", "nutrition", "rehabilitation"
    ],
    "education": [
        "online learning", "educational technology", "curriculum development",
        "instructional design", "assessment methods", "distance education",
        "blended learning", "personalized learning", "educational psychology",
        "cognitive science", "educational policy", "higher education", "K-12 education",
        "professional development", "lifelong learning", "skill acquisition",
        "educational research", "learning analytics", "knowledge management", "pedagogy"
    ]
}

# Advanced paragraph templates for each category to create more coherent documents
PARAGRAPH_TEMPLATES = {
    "technology": [
        "Recent advances in {topic} have revolutionized how we approach {related_topic}. Researchers at {organization} demonstrated that {detailed_concept} can improve performance by up to {number}%, particularly when combined with {another_concept}. This breakthrough has significant implications for {application_area}, where traditional methods have struggled with {challenge}.",
        
        "The integration of {topic} with {related_topic} presents unique opportunities for solving complex problems in {application_area}. According to a study published in {journal}, the {specific_technique} approach achieved state-of-the-art results on {benchmark}, outperforming previous methods by a substantial margin. The key innovation lies in how the system handles {technical_aspect}, effectively addressing the limitations of {previous_approach}.",
        
        "As {topic} continues to evolve, we're seeing increasing adoption across {industry} sectors. Companies like {company1} and {company2} have implemented {solution_type} solutions that leverage advanced {technology_type} techniques to optimize {business_process}. The economic impact of these technologies is estimated to reach ${large_number} billion by {future_year}, with particularly strong growth in {geographic_region}.",
        
        "The ethical implications of {topic} remain a critical consideration for researchers and practitioners alike. Issues surrounding {ethical_concern} and {another_ethical_concern} require careful attention, especially as these technologies become more integrated into {domain} systems that impact human lives. Several frameworks have been proposed to address these concerns, including {framework_name}, which emphasizes {principle} as a guiding principle.",
        
        "Looking ahead, the convergence of {topic} with {emerging_technology} is likely to open new frontiers in {research_area}. Early experiments combining these approaches have shown promising results for addressing {specific_challenge} that has long been considered a grand challenge in the field. Researchers at {research_institution} are currently exploring how {specific_technique} can be applied to {application} with potentially transformative outcomes."
    ],
    
    "science": [
        "Research in {topic} has yielded remarkable insights into {natural_phenomenon}. A team led by Dr. {scientist_name} at {institution} has recently published findings in {journal} demonstrating how {specific_mechanism} operates at the {scale} level. Their work challenges conventional understanding of {established_concept} by revealing that {surprising_finding}.",
        
        "The intersection of {topic} and {related_field} has become a fertile ground for scientific discovery. Using advanced {methodology}, scientists have been able to observe {phenomenon} with unprecedented precision, revealing {detailed_observation} that was previously impossible to detect. These findings suggest that {theoretical_implication}, which may require revising current models of {scientific_area}.",
        
        "Long-standing questions in {topic} are being addressed through innovative applications of {technique}. Recent experiments at {famous_lab} have successfully {experimental_achievement}, allowing researchers to test predictions made by {theory} under extreme conditions. The results confirm that {confirmed_hypothesis}, while raising intriguing questions about {unexplained_observation}.",
        
        "The environmental implications of {topic} extend far beyond what was initially anticipated. Data collected over {time_period} shows clear evidence of {environmental_impact}, particularly in {ecosystem_type} ecosystems. Scientists warn that without significant changes to {human_activity}, we may reach a tipping point for {critical_system} within {timeframe}, with potentially irreversible consequences.",
        
        "Collaborative efforts across {topic} and {another_field} are accelerating progress on complex problems like {complex_problem}. An international consortium of researchers has developed a {new_approach} that combines insights from multiple disciplines, leading to a {percentage}% improvement in {measurement_metric}. This interdisciplinary approach illustrates the value of breaking down traditional barriers between scientific fields."
    ],
    
    "business": [
        "The transformation of {industry} through {technology} represents one of the most significant business trends of the decade. Companies that have successfully implemented {specific_strategy} have seen their market share increase by {percentage}% on average, according to {consulting_firm}'s analysis. The key success factors include {factor1}, {factor2}, and a strong emphasis on {business_principle}.",
        
        "Investment in {emerging_sector} has reached unprecedented levels, with venture capital funding exceeding ${amount} billion in Q{quarter} of {year}. Startups focusing on {niche_area} are particularly attractive to investors due to their potential to disrupt {traditional_industry} with more efficient, {adjective} solutions. {startup_name}, which recently secured ${funding_amount} million in Series {round} funding, exemplifies this trend with its innovative approach to {business_problem}.",
        
        "Supply chain resilience has become a top priority for {industry} leaders following disruptions caused by {global_event}. Organizations are increasingly adopting {methodology} frameworks to better anticipate and mitigate risks across their value chains. A survey by {research_organization} found that companies with mature {risk_management} capabilities were {number} times more likely to maintain operational continuity during major disruptions.",
        
        "The shift toward sustainable business practices is reshaping competitive dynamics in {industry}. Companies that have embedded {sustainability_principle} into their core business model are not only reducing their environmental footprint but also realizing significant cost savings and enhancing brand equity. {company_name}'s initiative to {green_initiative} has resulted in a {percentage}% reduction in {resource} consumption while improving customer loyalty metrics by {points} points.",
        
        "Data-driven decision making has become a key differentiator for high-performing organizations across sectors. By leveraging advanced {analytics_type} analytics, businesses can {benefit} and gain {competitive_advantage}. However, successful implementation requires more than just technology—it demands a cultural shift toward {cultural_aspect} and investment in building {skill_type} capabilities throughout the organization."
    ],
    
    "healthcare": [
        "Advances in {medical_field} are transforming how we diagnose and treat {disease}. A clinical trial conducted at {medical_center} demonstrated that {treatment_approach} resulted in a {percentage}% improvement in {health_outcome} compared to standard care protocols. This promising approach combines {technology} with {methodology} to provide more personalized treatment options for patients, particularly those with {condition_characteristic}.",
        
        "The integration of {technology} into healthcare delivery systems has significant implications for patient outcomes and operational efficiency. A study published in {medical_journal} found that implementing {specific_system} reduced {negative_outcome} by {percentage}% while simultaneously improving provider satisfaction scores. However, challenges related to {challenge} and {another_challenge} must be addressed to realize the full potential of these technologies.",
        
        "Preventive healthcare strategies focusing on {health_aspect} have shown remarkable effectiveness in reducing the incidence of {chronic_condition}. Research by Dr. {researcher} and colleagues suggests that {intervention} can lower risk factors by up to {percentage}% when implemented consistently over {time_period}. These findings highlight the importance of shifting healthcare resources toward prevention rather than focusing primarily on treatment.",
        
        "The social determinants of health, including {determinant1} and {determinant2}, play a crucial role in health outcomes for {population} populations. A comprehensive analysis by {organization} revealed that addressing these factors through {policy_approach} could potentially reduce healthcare disparities by {percentage}% and generate savings of approximately ${amount} billion annually in avoidable healthcare costs.",
        
        "Ethical considerations in {healthcare_area} have become increasingly complex with the advent of {technology}. Healthcare institutions must navigate challenges related to {ethical_issue} while ensuring that innovations benefit all patients equitably. A framework developed by {ethics_committee} proposes {number} core principles to guide decision-making, emphasizing the importance of {principle} and {another_principle} in maintaining trust in healthcare systems."
    ],
    
    "education": [
        "Educational research on {learning_approach} has demonstrated significant benefits for student engagement and knowledge retention. A longitudinal study conducted across {number} schools found that implementing {specific_method} resulted in a {percentage}% improvement in {learning_outcome}, with particularly strong effects for {student_demographic}. These findings suggest that traditional {traditional_approach} may benefit from incorporating elements of this evidence-based approach.",
        
        "The application of {technology} in educational settings has transformed how students interact with {subject_area} content. Researchers at {university} developed a {tool_name} that utilizes {specific_technology} to provide personalized learning experiences, addressing the diverse needs of learners with different {learning_characteristic}. Early adoption in {number} classrooms shows promising results, with {percentage}% of students reporting enhanced motivation and deeper understanding of complex concepts.",
        
        "Assessment methodologies in {educational_level} education are evolving to better measure {skill_type} skills that are increasingly valued in today's knowledge economy. Moving beyond traditional {traditional_assessment}, innovative approaches like {assessment_method} provide more authentic evaluation of students' ability to {complex_task}. Evidence suggests these methods not only assess learning more accurately but also reinforce the development of {21st_century_skill} that students will need for future success.",
        
        "Professional development programs for educators focusing on {teaching_approach} have shown remarkable effectiveness in improving instructional quality. A study involving {number} teachers across {location} found that intensive training in {specific_pedagogy} led to measurable improvements in student outcomes, including a {percentage}% increase in {metric}. The most successful programs included elements of {component1}, {component2}, and ongoing support through {support_mechanism}.",
        
        "Educational equity remains a critical challenge, particularly in terms of access to {resource} and opportunities for {disadvantaged_group} students. Research by {organization} highlights how {intervention} initiatives can help close achievement gaps when implemented with fidelity and sufficient resources. Key success factors include {factor1}, {factor2}, and meaningful engagement with {stakeholder} throughout the process."
    ]
}

# Common entities to make texts more realistic
ORGANIZATIONS = ["Stanford University", "MIT", "Google Research", "Microsoft Research", "DeepMind", 
                "OpenAI", "Facebook AI Research", "IBM Research", "Harvard University", "Berkeley Lab",
                "Tsinghua University", "ETH Zurich", "Max Planck Institute", "CERN", "NASA",
                "AllenAI", "EPFL", "University of Toronto", "Carnegie Mellon University", "Oxford University"]

JOURNALS = ["Nature", "Science", "Cell", "PNAS", "JAMA", "The Lancet", "IEEE Transactions", 
            "ACM Computing Surveys", "Journal of Machine Learning Research", "Proceedings of the National Academy of Sciences",
            "New England Journal of Medicine", "Chemical Reviews", "Physical Review Letters", "Quarterly Journal of Economics",
            "Journal of Finance", "Academy of Management Journal", "Harvard Business Review", "Educational Researcher"]

PEOPLE = ["James Johnson", "Maria Rodriguez", "Wei Zhang", "Aisha Patel", "David Kim", 
          "Sophia Nguyen", "Carlos Mendoza", "Emma Thompson", "Omar Hassan", "Yuki Tanaka",
          "Alexander Schmidt", "Fatima Al-Fihri", "Roberto Bianchi", "Priya Sharma", "John Smith",
          "Olga Petrov", "Jamal Washington", "Mei Lin", "Henrik Johansson", "Chioma Okonkwo"]

LOCATIONS = ["San Francisco", "Shanghai", "London", "Mumbai", "Tokyo", "Berlin", "Singapore", 
             "New York", "Stockholm", "Geneva", "Seoul", "Bangalore", "Boston", "Tel Aviv", 
             "Paris", "Toronto", "Zurich", "Helsinki", "Sydney", "Dubai"]

COMPANIES = ["Alphabet", "Microsoft", "Apple", "Amazon", "Meta", "Tesla", "IBM", "Intel", 
             "NVIDIA", "Samsung", "Siemens", "Toyota", "Johnson & Johnson", "Pfizer", "Roche",
             "JP Morgan Chase", "Goldman Sachs", "Alibaba", "Tencent", "Salesforce"]

# Helper functions
def fill_template(template: str, category: str) -> str:
    """Fill template with category-specific content."""
    # Extract placeholders
    placeholders = []
    current = ""
    inside_placeholder = False
    
    for char in template:
        if char == '{':
            inside_placeholder = True
            current = ""
        elif char == '}' and inside_placeholder:
            inside_placeholder = False
            placeholders.append(current)
        elif inside_placeholder:
            current += char
    
    # Fill placeholders with appropriate content
    filled_template = template
    for placeholder in placeholders:
        replacement = ""
        
        # Generate appropriate content based on placeholder type
        if "topic" in placeholder or "field" in placeholder or "area" in placeholder:
            replacement = random.choice(CATEGORIES[category])
        elif "organization" in placeholder or "institution" in placeholder:
            replacement = random.choice(ORGANIZATIONS)
        elif "journal" in placeholder:
            replacement = random.choice(JOURNALS)
        elif "name" in placeholder or "scientist" in placeholder or "researcher" in placeholder:
            replacement = random.choice(PEOPLE)
        elif "company" in placeholder:
            replacement = random.choice(COMPANIES)
        elif "location" in placeholder:
            replacement = random.choice(LOCATIONS)
        elif "percentage" in placeholder:
            replacement = str(random.randint(15, 85))
        elif "number" in placeholder:
            replacement = str(random.randint(2, 50))
        elif "amount" in placeholder or "large_number" in placeholder:
            replacement = str(random.randint(10, 500))
        elif "year" in placeholder or "future_year" in placeholder:
            replacement = str(random.randint(2024, 2035))
        else:
            # Generic replacement for other placeholders
            replacement = f"example {placeholder}"
        
        filled_template = filled_template.replace(f"{{{placeholder}}}", replacement)
    
    return filled_template

def generate_document(category: str, paragraph_count: int = 5) -> str:
    """Generate a document for a specific category with multiple paragraphs."""
    paragraphs = []
    
    # Create an introductory paragraph
    templates = PARAGRAPH_TEMPLATES[category]
    intro_template = random.choice(templates)
    intro_paragraph = fill_template(intro_template, category)
    paragraphs.append(intro_paragraph)
    
    # Add content paragraphs
    for _ in range(paragraph_count - 1):
        template = random.choice(templates)
        paragraph = fill_template(template, category)
        paragraphs.append(paragraph)
    
    # Join paragraphs with line breaks
    document = "\n\n".join(paragraphs)
    return document

def generate_query(category: str, document: str) -> Tuple[str, str]:
    """Generate a search query and expected answer based on document content."""
    # Extract some key terms from the document
    words = document.split()
    keywords = []
    
    # Find some domain-specific terms
    for term in CATEGORIES[category]:
        if term.lower() in document.lower():
            keywords.append(term)
    
    if not keywords:
        # Fallback if no category terms found
        keywords = random.sample(CATEGORIES[category], 2)
    
    # Generate query templates
    query_templates = [
        "What are the latest developments in {topic}?",
        "How does {topic} impact {related_topic}?",
        "Explain the relationship between {topic} and {related_topic}",
        "What are the key benefits of {topic} in {industry}?",
        "How is {organization} using {topic} to advance research?",
        "What challenges exist in implementing {topic}?",
        "Compare {topic} with traditional approaches in {field}",
        "What future trends are expected in {topic}?",
        "How can {topic} be applied to solve problems in {application}?",
        "What ethical considerations are important for {topic}?"
    ]
    
    # Select a query template and fill it
    template = random.choice(query_templates)
    query = template.format(
        topic=random.choice(keywords),
        related_topic=random.choice(CATEGORIES[category]),
        organization=random.choice(ORGANIZATIONS),
        field=random.choice(CATEGORIES[category]),
        industry=random.choice(["healthcare", "finance", "manufacturing", "retail", "education"]),
        application=random.choice(["business", "science", "healthcare", "education"]),
    )
    
    # Extract a relevant passage as the expected answer
    paragraphs = document.split("\n\n")
    most_relevant_paragraph = ""
    max_relevance = -1
    
    for paragraph in paragraphs:
        relevance = 0
        for term in query.lower().split():
            if len(term) > 3 and term.lower() in paragraph.lower():
                relevance += 1
        
        if relevance > max_relevance:
            max_relevance = relevance
            most_relevant_paragraph = paragraph
    
    return query, most_relevant_paragraph

def generate_dataset(
    num_documents: int = 1000,
    num_queries_per_doc: int = 2,
    paragraphs_per_doc: int = 5,
    output_dir: str = "enhanced_data"
) -> None:
    """Generate a dataset of documents and queries."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    documents = []
    queries = []
    
    print(f"Generating {num_documents} documents...")
    for doc_id in tqdm(range(num_documents)):
        # Choose a random category
        category = random.choice(list(CATEGORIES.keys()))
        
        # Generate a document
        document = generate_document(category, paragraph_count=paragraphs_per_doc)
        
        # Generate queries for this document
        doc_queries = []
        for _ in range(num_queries_per_doc):
            query, answer = generate_query(category, document)
            doc_queries.append({
                "query": query,
                "answer": answer,
                "document_id": doc_id
            })
        
        # Add to dataset
        documents.append({
            "id": doc_id,
            "category": category,
            "text": document
        })
        queries.extend(doc_queries)
    
    # Save documents
    with open(os.path.join(output_dir, "documents.json"), "w") as f:
        json.dump(documents, f, indent=2)
    
    # Save queries
    with open(os.path.join(output_dir, "queries.json"), "w") as f:
        json.dump(queries, f, indent=2)
    
    # Create a sample text file with all documents for embedding testing
    with open(os.path.join(output_dir, "all_documents.txt"), "w") as f:
        for doc in documents:
            f.write(f"Document {doc['id']} ({doc['category']}):\n")
            f.write(doc['text'])
            f.write("\n\n" + "-"*80 + "\n\n")
    
    # Save dataset stats
    stats = {
        "total_documents": len(documents),
        "total_queries": len(queries),
        "queries_per_document": num_queries_per_doc,
        "paragraphs_per_document": paragraphs_per_doc,
        "category_distribution": {cat: 0 for cat in CATEGORIES.keys()}
    }
    
    for doc in documents:
        stats["category_distribution"][doc["category"]] += 1
    
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset generated successfully in '{output_dir}'")
    print(f"  - {len(documents)} documents")
    print(f"  - {len(queries)} queries")
    print(f"  - Categories: {', '.join(f'{k}: {v}' for k, v in stats['category_distribution'].items())}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate enhanced sample data for dimensional cascade testing")
    parser.add_argument("--num-documents", type=int, default=1000, help="Number of documents to generate")
    parser.add_argument("--queries-per-doc", type=int, default=2, help="Number of queries per document")
    parser.add_argument("--paragraphs-per-doc", type=int, default=5, help="Number of paragraphs per document")
    parser.add_argument("--output-dir", type=str, default="enhanced_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Generating enhanced sample data for dimensional cascade testing")
    print(f"  - Number of documents: {args.num_documents}")
    print(f"  - Queries per document: {args.queries_per_doc}")
    print(f"  - Paragraphs per document: {args.paragraphs_per_doc}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Random seed: {args.seed}")
    
    generate_dataset(
        num_documents=args.num_documents,
        num_queries_per_doc=args.queries_per_doc,
        paragraphs_per_doc=args.paragraphs_per_doc,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 