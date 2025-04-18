#!/usr/bin/env python
# Generate sample data for testing dimensional cascade

import json
import argparse
import random
import os
from typing import List, Dict, Any

# Categories and their keywords for generating documents
CATEGORIES = {
    "technology": [
        "computer", "software", "hardware", "internet", "programming", 
        "artificial intelligence", "machine learning", "data science",
        "algorithm", "database", "cloud computing", "cybersecurity",
        "automation", "blockchain", "virtual reality", "network"
    ],
    "health": [
        "medicine", "doctor", "hospital", "patient", "disease",
        "treatment", "therapy", "diagnosis", "symptom", "healthcare",
        "wellness", "nutrition", "fitness", "surgery", "pharmaceutical"
    ],
    "business": [
        "company", "corporation", "startup", "entrepreneur", "investment",
        "finance", "market", "strategy", "management", "leadership",
        "profit", "revenue", "customer", "product", "service", "innovation"
    ],
    "science": [
        "research", "experiment", "laboratory", "theory", "hypothesis",
        "scientist", "physics", "chemistry", "biology", "astronomy",
        "mathematics", "engineering", "discovery", "observation", "analysis"
    ],
    "arts": [
        "music", "painting", "sculpture", "theater", "literature",
        "poetry", "novel", "artist", "creativity", "performance",
        "exhibition", "gallery", "museum", "culture", "aesthetic"
    ]
}

# Sample sentences for each category to make more coherent documents
SAMPLE_SENTENCES = {
    "technology": [
        "The latest advances in technology have revolutionized our daily lives.",
        "Software developers are creating innovative solutions to complex problems.",
        "Artificial intelligence is transforming industries across the globe.",
        "Machine learning algorithms can analyze vast amounts of data quickly.",
        "Cloud computing offers scalable resources for businesses of all sizes.",
        "Cybersecurity threats continue to evolve at an alarming rate.",
        "Programming languages are becoming more accessible to beginners.",
        "Virtual reality creates immersive experiences for users.",
        "Blockchain technology ensures secure and transparent transactions.",
        "The Internet of Things connects everyday devices to the internet."
    ],
    "health": [
        "Regular exercise is essential for maintaining good health.",
        "Doctors recommend a balanced diet rich in fruits and vegetables.",
        "Hospitals are implementing new patient care protocols.",
        "Medical researchers are developing innovative treatments for diseases.",
        "Mental health awareness has increased significantly in recent years.",
        "Preventive healthcare measures can reduce the risk of chronic illnesses.",
        "Telemedicine has made healthcare more accessible to remote areas.",
        "Nutrition plays a crucial role in overall wellness and longevity.",
        "Healthcare professionals work tirelessly to improve patient outcomes.",
        "Advancements in medical technology have led to more accurate diagnoses."
    ],
    "business": [
        "Successful entrepreneurs identify market gaps and create solutions.",
        "Innovative business models are disrupting traditional industries.",
        "Strategic planning is essential for long-term business success.",
        "Effective leadership inspires teams to achieve organizational goals.",
        "Market analysis helps companies understand consumer preferences.",
        "Financial management ensures sustainable business operations.",
        "Customer satisfaction drives business growth and loyalty.",
        "Corporate social responsibility initiatives benefit communities.",
        "Digital transformation is reshaping business processes globally.",
        "Startups are attracting significant investment despite economic challenges."
    ],
    "science": [
        "Scientific research advances our understanding of the natural world.",
        "Experiments in laboratories test hypotheses and generate new knowledge.",
        "Theoretical physics explores the fundamental laws of the universe.",
        "Biologists study living organisms and their interactions with ecosystems.",
        "Chemistry research leads to the development of new materials and compounds.",
        "Astronomical observations reveal the mysteries of distant galaxies.",
        "Mathematical models help scientists predict natural phenomena.",
        "Engineering principles are applied to solve practical problems.",
        "Environmental science addresses challenges related to climate change.",
        "Interdisciplinary research combines insights from multiple scientific fields."
    ],
    "arts": [
        "Music has the power to evoke deep emotions and connect people.",
        "Visual artists express ideas through various mediums and techniques.",
        "Literary works reflect societal values and human experiences.",
        "Performing arts bring stories to life through movement and expression.",
        "Art galleries showcase diverse perspectives and creative visions.",
        "Cultural heritage is preserved through artistic traditions.",
        "Creative processes involve both inspiration and disciplined practice.",
        "Artistic movements emerge in response to historical contexts.",
        "Digital technologies have expanded possibilities for artistic creation.",
        "Art education nurtures creativity and critical thinking skills."
    ]
}

def generate_document(category: str, word_count: int = 200) -> str:
    """Generate a document for a specific category."""
    keywords = CATEGORIES[category]
    sentences = SAMPLE_SENTENCES[category]
    
    # Start with some sentences from the category
    doc_sentences = random.sample(sentences, min(3, len(sentences)))
    
    # Add random sentences until we reach the desired word count
    word_total = sum(len(s.split()) for s in doc_sentences)
    
    while word_total < word_count:
        if random.random() < 0.7:  # 70% chance to use a sentence from the category
            sentence = random.choice(sentences)
        else:  # 30% chance to create a new sentence with keywords
            num_keywords = random.randint(1, 3)
            selected_keywords = random.sample(keywords, num_keywords)
            
            templates = [
                "The {} is an important aspect to consider.",
                "Many experts focus on {} in their research.",
                "Recent developments in {} have shown promising results.",
                "Understanding {} can lead to significant breakthroughs.",
                "The role of {} cannot be underestimated.",
                "Advances in {} continue to shape the field."
            ]
            
            template = random.choice(templates)
            sentence = template.format(" and ".join(selected_keywords))
        
        doc_sentences.append(sentence)
        word_total += len(sentence.split())
    
    return " ".join(doc_sentences)

def generate_query(category: str) -> str:
    """Generate a search query for a specific category."""
    keywords = CATEGORIES[category]
    
    templates = [
        "What are the latest developments in {}?",
        "How does {} impact society?",
        "The importance of {} in modern context",
        "Why is {} significant?",
        "Benefits of {} for individuals",
        "The relationship between {} and progress",
        "Understanding {} better",
        "The future of {}"
    ]
    
    template = random.choice(templates)
    keyword = random.choice(keywords)
    
    return template.format(keyword)

def generate_sample_data(
    num_documents: int = 1000,
    num_queries: int = 50,
    output_dir: str = "sample_data"
) -> None:
    """Generate sample documents and queries."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate documents
    documents = []
    document_categories = []
    
    for i in range(num_documents):
        category = random.choice(list(CATEGORIES.keys()))
        document = generate_document(category)
        documents.append(document)
        document_categories.append(category)
    
    # Save documents
    with open(os.path.join(output_dir, "documents.json"), "w") as f:
        json.dump(documents, f, indent=2)
    
    # Save document metadata (including categories)
    document_metadata = [{"id": i, "category": cat} for i, cat in enumerate(document_categories)]
    with open(os.path.join(output_dir, "document_metadata.json"), "w") as f:
        json.dump(document_metadata, f, indent=2)
    
    # Generate queries
    queries = []
    query_categories = []
    
    for i in range(num_queries):
        category = random.choice(list(CATEGORIES.keys()))
        query = generate_query(category)
        queries.append(query)
        query_categories.append(category)
    
    # Save queries
    with open(os.path.join(output_dir, "queries.json"), "w") as f:
        json.dump(queries, f, indent=2)
    
    # Save query metadata
    query_metadata = [{"id": i, "category": cat} for i, cat in enumerate(query_categories)]
    with open(os.path.join(output_dir, "query_metadata.json"), "w") as f:
        json.dump(query_metadata, f, indent=2)
    
    print(f"Generated {num_documents} documents and {num_queries} queries in {output_dir}")
    print(f"Category distribution (documents):")
    category_counts = {}
    for cat in document_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} ({count/num_documents*100:.1f}%)")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sample data for dimensional cascade')
    parser.add_argument('--num_documents', type=int, default=1000,
                        help='Number of documents to generate')
    parser.add_argument('--num_queries', type=int, default=50,
                        help='Number of queries to generate')
    parser.add_argument('--output_dir', type=str, default='sample_data',
                        help='Directory to save generated data')
    return parser.parse_args()

def main():
    args = parse_args()
    generate_sample_data(
        num_documents=args.num_documents,
        num_queries=args.num_queries,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 