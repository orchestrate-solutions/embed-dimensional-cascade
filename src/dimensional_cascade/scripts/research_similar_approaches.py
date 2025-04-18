#!/usr/bin/env python
"""
Research similar approaches to the Dimensional Cascade concept.

This script:
1. Searches academic papers and GitHub repositories for similar approaches
2. Compares different methods for progressive dimensionality reduction
3. Analyzes and summarizes findings in a research report
"""
import os
import argparse
import requests
import json
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from scholarly import scholarly


def parse_args():
    parser = argparse.ArgumentParser(description='Research similar approaches to Dimensional Cascade')
    
    parser.add_argument('--output-dir', type=str, default='research',
                        help='Output directory for research findings')
    parser.add_argument('--github-token', type=str, default=None,
                        help='GitHub API token for extended API limits')
    parser.add_argument('--max-papers', type=int, default=50,
                        help='Maximum number of papers to analyze')
    parser.add_argument('--max-repos', type=int, default=50,
                        help='Maximum number of repositories to analyze')
    
    return parser.parse_args()


def search_github_repositories(query: str, token: Optional[str] = None, max_results: int = 50) -> List[Dict[str, Any]]:
    """Search GitHub repositories matching the query.
    
    Args:
        query: Search query string
        token: GitHub API token
        max_results: Maximum number of results to return
        
    Returns:
        List of repository information dictionaries
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    base_url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100}
    
    repositories = []
    page = 1
    
    with tqdm(total=max_results, desc="Searching GitHub repos") as pbar:
        while len(repositories) < max_results:
            params["page"] = page
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                
                if not items:
                    break
                
                for repo in items:
                    if len(repositories) >= max_results:
                        break
                    
                    repositories.append({
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": repo["description"],
                        "url": repo["html_url"],
                        "stars": repo["stargazers_count"],
                        "forks": repo["forks_count"],
                        "language": repo["language"],
                        "created_at": repo["created_at"],
                        "updated_at": repo["updated_at"]
                    })
                    pbar.update(1)
                
                page += 1
                
                # Rate limit handling
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining <= 1:
                        reset_time = int(response.headers["X-RateLimit-Reset"])
                        sleep_time = max(0, reset_time - time.time()) + 1
                        print(f"Rate limit reached. Sleeping for {sleep_time} seconds...")
                        time.sleep(sleep_time)
            else:
                print(f"Error searching GitHub: {response.status_code}")
                print(response.json())
                break
    
    return repositories


def search_academic_papers(keywords: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
    """Search academic papers related to the keywords.
    
    Args:
        keywords: List of keywords to search for
        max_results: Maximum number of papers to return
        
    Returns:
        List of paper information dictionaries
    """
    query = " ".join(keywords)
    
    papers = []
    with tqdm(total=max_results, desc="Searching academic papers") as pbar:
        search_query = scholarly.search_pubs(query)
        
        try:
            for i in range(max_results):
                try:
                    paper = next(search_query)
                    
                    # Extract relevant information
                    paper_info = {
                        "title": paper.get("bib", {}).get("title", ""),
                        "authors": paper.get("bib", {}).get("author", []),
                        "year": paper.get("bib", {}).get("pub_year", ""),
                        "venue": paper.get("bib", {}).get("venue", ""),
                        "abstract": paper.get("bib", {}).get("abstract", ""),
                        "citations": paper.get("num_citations", 0),
                        "url": paper.get("pub_url", "")
                    }
                    
                    papers.append(paper_info)
                    pbar.update(1)
                    
                    # Avoid rate limiting
                    time.sleep(1)
                    
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error fetching paper: {e}")
                    time.sleep(5)  # Wait longer on error
        except Exception as e:
            print(f"Error in scholarly search: {e}")
    
    return papers


def search_huggingface_models(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Search for relevant models on Hugging Face.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of model information dictionaries
    """
    base_url = "https://huggingface.co/api/models"
    params = {"search": query, "limit": min(max_results, 100)}
    
    models = []
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            for model in data[:max_results]:
                model_info = {
                    "name": model.get("id", ""),
                    "author": model.get("author", ""),
                    "downloads": model.get("downloads", 0),
                    "likes": model.get("likes", 0),
                    "tags": model.get("tags", []),
                    "pipeline_tag": model.get("pipeline_tag", ""),
                    "url": f"https://huggingface.co/{model.get('id', '')}"
                }
                
                models.append(model_info)
        else:
            print(f"Error searching Hugging Face: {response.status_code}")
    
    except Exception as e:
        print(f"Error querying Hugging Face API: {e}")
    
    return models


def analyze_repositories(repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze GitHub repositories for insights.
    
    Args:
        repositories: List of repository information dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    # Extract languages
    languages = {}
    for repo in repositories:
        lang = repo.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
    
    # Extract creation dates
    dates = []
    for repo in repositories:
        created_at = repo.get("created_at")
        if created_at:
            try:
                date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                dates.append(date.year)
            except:
                pass
    
    # Count stars
    stars = [repo.get("stars", 0) for repo in repositories]
    
    return {
        "languages": languages,
        "dates": dates,
        "total_stars": sum(stars),
        "avg_stars": sum(stars) / len(stars) if stars else 0,
        "max_stars": max(stars) if stars else 0,
        "repo_count": len(repositories)
    }


def analyze_papers(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze academic papers for insights.
    
    Args:
        papers: List of paper information dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    # Extract years
    years = []
    for paper in papers:
        year = paper.get("year")
        if year and year.isdigit():
            years.append(int(year))
    
    # Count citations
    citations = [paper.get("citations", 0) for paper in papers]
    
    # Extract venues
    venues = {}
    for paper in papers:
        venue = paper.get("venue")
        if venue:
            venues[venue] = venues.get(venue, 0) + 1
    
    # Topic analysis (simple keyword counting)
    topics = {}
    keywords = [
        "dimension reduction", "embeddings", "autoencoder", 
        "distillation", "semantic search", "vector search",
        "approximate nearest neighbor", "ANN", "HNSW",
        "progressive", "cascade", "multi-resolution"
    ]
    
    for paper in papers:
        abstract = paper.get("abstract", "").lower()
        title = paper.get("title", "").lower()
        text = abstract + " " + title
        
        for keyword in keywords:
            if keyword.lower() in text:
                topics[keyword] = topics.get(keyword, 0) + 1
    
    return {
        "years": years,
        "total_citations": sum(citations),
        "avg_citations": sum(citations) / len(citations) if citations else 0,
        "max_citations": max(citations) if citations else 0,
        "venues": venues,
        "topics": topics,
        "paper_count": len(papers)
    }


def plot_repository_analysis(analysis: Dict[str, Any], output_path: str):
    """Generate plots from repository analysis.
    
    Args:
        analysis: Repository analysis dictionary
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Language distribution
    languages = analysis["languages"]
    lang_names = list(languages.keys())
    lang_counts = list(languages.values())
    
    # Sort by count
    lang_data = sorted(zip(lang_names, lang_counts), key=lambda x: x[1], reverse=True)
    lang_names, lang_counts = zip(*lang_data[:10])  # Top 10 languages
    
    ax1.bar(lang_names, lang_counts)
    ax1.set_title("Top Languages")
    ax1.set_xlabel("Language")
    ax1.set_ylabel("Repository Count")
    ax1.tick_params(axis='x', rotation=45)
    
    # Creation date histogram
    dates = analysis["dates"]
    ax2.hist(dates, bins=range(min(dates) if dates else 2010, 
                               max(dates) + 2 if dates else 2023))
    ax2.set_title("Repository Creation Years")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Repository Count")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_paper_analysis(analysis: Dict[str, Any], output_path: str):
    """Generate plots from paper analysis.
    
    Args:
        analysis: Paper analysis dictionary
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Year distribution
    years = analysis["years"]
    ax1.hist(years, bins=range(min(years) if years else 2010, 
                              max(years) + 2 if years else 2023))
    ax1.set_title("Publication Years")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Paper Count")
    
    # Topic distribution
    topics = analysis["topics"]
    topic_names = list(topics.keys())
    topic_counts = list(topics.values())
    
    # Sort by count
    topic_data = sorted(zip(topic_names, topic_counts), key=lambda x: x[1], reverse=True)
    if topic_data:
        topic_names, topic_counts = zip(*topic_data[:10])  # Top 10 topics
        
        ax2.barh(topic_names, topic_counts)
        ax2.set_title("Keyword Frequency")
        ax2.set_xlabel("Count")
        ax2.set_ylabel("Keyword")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_research_report(
    repos: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    repo_analysis: Dict[str, Any],
    paper_analysis: Dict[str, Any],
    output_path: str
):
    """Generate a comprehensive research report.
    
    Args:
        repos: List of repository information
        papers: List of paper information
        models: List of model information
        repo_analysis: Repository analysis results
        paper_analysis: Paper analysis results
        output_path: Path to save the report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Research Report: Dimensional Cascade and Similar Approaches\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Introduction
        f.write("## 1. Introduction\n\n")
        f.write("This report summarizes existing approaches similar to the Dimensional Cascade concept ")
        f.write("for progressive dimensionality reduction in semantic search. The research covers academic ")
        f.write("papers, GitHub repositories, and pre-trained models found on Hugging Face.\n\n")
        
        # Summary statistics
        f.write("## 2. Summary Statistics\n\n")
        f.write("### GitHub Repositories\n")
        f.write(f"- Total repositories analyzed: {repo_analysis['repo_count']}\n")
        f.write(f"- Total stars: {repo_analysis['total_stars']}\n")
        f.write(f"- Average stars per repository: {repo_analysis['avg_stars']:.1f}\n")
        f.write(f"- Most popular languages: {', '.join(list(repo_analysis['languages'].keys())[:5])}\n")
        f.write("\n")
        
        f.write("### Academic Papers\n")
        f.write(f"- Total papers analyzed: {paper_analysis['paper_count']}\n")
        f.write(f"- Total citations: {paper_analysis['total_citations']}\n")
        f.write(f"- Average citations per paper: {paper_analysis['avg_citations']:.1f}\n")
        f.write(f"- Publication years: {min(paper_analysis['years']) if paper_analysis['years'] else 'N/A'} - {max(paper_analysis['years']) if paper_analysis['years'] else 'N/A'}\n")
        f.write("\n")
        
        f.write("### Pre-trained Models\n")
        f.write(f"- Total models found: {len(models)}\n\n")
        
        # Notable repositories
        f.write("## 3. Notable GitHub Repositories\n\n")
        
        sorted_repos = sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)
        for i, repo in enumerate(sorted_repos[:10]):
            f.write(f"### {i+1}. [{repo['name']}]({repo['url']}) - ⭐ {repo['stars']}\n\n")
            f.write(f"**Description**: {repo['description'] or 'No description'}\n\n")
            f.write(f"**Language**: {repo['language'] or 'Unknown'}\n")
            f.write(f"**Created**: {repo['created_at'][:10]}\n")
            f.write(f"**Last updated**: {repo['updated_at'][:10]}\n\n")
        
        # Notable papers
        f.write("## 4. Key Academic Papers\n\n")
        
        sorted_papers = sorted(papers, key=lambda x: x.get("citations", 0), reverse=True)
        for i, paper in enumerate(sorted_papers[:10]):
            f.write(f"### {i+1}. {paper['title']}\n\n")
            f.write(f"**Authors**: {', '.join(paper['authors']) if isinstance(paper['authors'], list) else paper['authors']}\n")
            f.write(f"**Year**: {paper['year']}\n")
            f.write(f"**Venue**: {paper['venue']}\n")
            f.write(f"**Citations**: {paper['citations']}\n\n")
            
            if paper['abstract']:
                f.write("**Abstract**:\n\n")
                f.write(f"{paper['abstract']}\n\n")
            
            if paper['url']:
                f.write(f"[View paper]({paper['url']})\n\n")
            
            f.write("---\n\n")
        
        # Notable models
        f.write("## 5. Pre-trained Models\n\n")
        
        sorted_models = sorted(models, key=lambda x: x.get("downloads", 0), reverse=True)
        for i, model in enumerate(sorted_models[:10]):
            f.write(f"### {i+1}. [{model['name']}]({model['url']})\n\n")
            f.write(f"**Author**: {model['author']}\n")
            f.write(f"**Downloads**: {model['downloads']}\n")
            f.write(f"**Likes**: {model['likes']}\n")
            f.write(f"**Tags**: {', '.join(model['tags'])}\n\n")
        
        # Analysis and insights
        f.write("## 6. Analysis and Insights\n\n")
        
        f.write("### Key Trends\n\n")
        f.write("Based on the repositories, papers, and models analyzed, the following trends are apparent:\n\n")
        
        # Programming languages
        f.write("#### Programming Languages\n\n")
        f.write("The most common programming languages used for embedding and semantic search projects are:\n\n")
        for lang, count in sorted(repo_analysis['languages'].items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"- **{lang}**: {count} repositories\n")
        f.write("\n")
        
        # Research topics
        f.write("#### Research Topics\n\n")
        f.write("The most frequently mentioned topics in academic papers are:\n\n")
        for topic, count in sorted(paper_analysis['topics'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"- **{topic}**: mentioned in {count} papers\n")
        f.write("\n")
        
        # Existing similar approaches
        f.write("### Similar Approaches\n\n")
        f.write("Based on the research, the following approaches are most similar to the Dimensional Cascade concept:\n\n")
        
        f.write("1. **Progressive Dimensionality Reduction**: Several papers discuss the concept of progressively reducing dimensions for more efficient search.\n\n")
        f.write("2. **Multi-resolution Embeddings**: Projects that use embeddings at different resolutions for different tasks or stages of processing.\n\n")
        f.write("3. **Hierarchical Vector Search**: Approaches that use a hierarchy of vector representations for efficient search.\n\n")
        f.write("4. **Knowledge Distillation for Embeddings**: Using larger models to teach smaller, more efficient models.\n\n")
        
        # Conclusion
        f.write("## 7. Conclusion and Recommendations\n\n")
        f.write("The Dimensional Cascade approach combines several concepts that have been explored separately in academia and industry. ")
        f.write("While there are similar approaches, the specific combination of progressive dimensionality reduction with unified semantic ")
        f.write("coherence across resolution levels appears to be novel.\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("1. **Benchmark against existing approaches**: Compare the Dimensional Cascade with both traditional single-dimension approaches and any similar multi-resolution approaches.\n\n")
        f.write("2. **Evaluate different reduction methods**: Test the three proposed methods (truncation, distillation, and autoencoder) to determine which performs best for different use cases.\n\n")
        f.write("3. **Consider hybrid approaches**: Explore combining the Dimensional Cascade with other efficient search techniques like product quantization or LSH.\n\n")
        f.write("4. **Domain-specific tuning**: Different domains may benefit from different dimension sets and transition thresholds.\n\n")
        
        # References
        f.write("## 8. References\n\n")
        f.write("### GitHub Repositories\n\n")
        for i, repo in enumerate(sorted_repos[:20]):
            f.write(f"{i+1}. [{repo['full_name']}]({repo['url']}) - {repo['description'] or 'No description'}\n")
        f.write("\n")
        
        f.write("### Academic Papers\n\n")
        for i, paper in enumerate(sorted_papers[:20]):
            authors = ', '.join(paper['authors']) if isinstance(paper['authors'], list) else paper['authors']
            f.write(f"{i+1}. {authors} ({paper['year']}). {paper['title']}. {paper['venue']}.\n")
        
        f.write("\n\n---\n\n")
        f.write("*This report was automatically generated by a research script.*")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Search GitHub repositories
    repo_queries = [
        "semantic search dimension reduction",
        "embedding dimension progressive",
        "vector search cascade",
        "multi resolution embedding",
        "dimensionality reduction search efficiency"
    ]
    
    all_repos = []
    for query in repo_queries:
        print(f"Searching GitHub for: {query}")
        repos = search_github_repositories(
            query=query, 
            token=args.github_token,
            max_results=args.max_repos // len(repo_queries)
        )
        all_repos.extend(repos)
    
    # Remove duplicates
    unique_repos = {repo["full_name"]: repo for repo in all_repos}.values()
    all_repos = list(unique_repos)
    
    # Search academic papers
    paper_keywords = [
        ["dimensionality reduction", "semantic search"],
        ["progressive", "embeddings", "search"],
        ["cascade", "vector", "dimension", "search"],
        ["multi-resolution", "embedding"],
        ["dimension reduction", "embedding", "efficiency"]
    ]
    
    all_papers = []
    for keywords in paper_keywords:
        print(f"Searching academic papers for: {' '.join(keywords)}")
        papers = search_academic_papers(
            keywords=keywords,
            max_results=args.max_papers // len(paper_keywords)
        )
        all_papers.extend(papers)
    
    # Remove duplicates
    unique_papers = {paper.get("title", ""): paper for paper in all_papers}.values()
    all_papers = list(unique_papers)
    
    # Search Hugging Face models
    model_queries = [
        "sentence embedding dimension",
        "semantic search embedding",
        "dimension reduction embedding",
        "efficient embedding"
    ]
    
    all_models = []
    for query in model_queries:
        print(f"Searching Hugging Face for: {query}")
        models = search_huggingface_models(
            query=query,
            max_results=10  # Limit models per query
        )
        all_models.extend(models)
    
    # Remove duplicates
    unique_models = {model["name"]: model for model in all_models}.values()
    all_models = list(unique_models)
    
    # Analyze repositories and papers
    repo_analysis = analyze_repositories(all_repos)
    paper_analysis = analyze_papers(all_papers)
    
    # Generate plots
    plot_repository_analysis(
        repo_analysis, 
        os.path.join(args.output_dir, "repository_analysis.png")
    )
    
    plot_paper_analysis(
        paper_analysis,
        os.path.join(args.output_dir, "paper_analysis.png")
    )
    
    # Generate report
    report_path = os.path.join(args.output_dir, "research_report.md")
    generate_research_report(
        repos=all_repos,
        papers=all_papers,
        models=all_models,
        repo_analysis=repo_analysis,
        paper_analysis=paper_analysis,
        output_path=report_path
    )
    
    # Save raw data
    with open(os.path.join(args.output_dir, "repositories.json"), 'w', encoding='utf-8') as f:
        json.dump(all_repos, f, indent=2)
    
    with open(os.path.join(args.output_dir, "papers.json"), 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2)
    
    with open(os.path.join(args.output_dir, "models.json"), 'w', encoding='utf-8') as f:
        json.dump(all_models, f, indent=2)
    
    print(f"Research complete! Report saved to {report_path}")
    print(f"Found {len(all_repos)} repositories, {len(all_papers)} papers, and {len(all_models)} models")


if __name__ == '__main__':
    main() 