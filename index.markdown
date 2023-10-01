---
layout: default
---

## Interests

- Interest 1
- Interest 2
- Interest 3

## Pet Projects

{% for project in site.pages %}
  {% if project.layout == 'project' %}
    ### {{ project.title }}
    - **Description**: {{ project.description }}
    - **Link**: [{{ project.title }} Link]({{ project.link }})
  {% endif %}
  {% if forloop.index == 4 %}
    {% break %}
  {% endif %}
{% endfor %}

## Recent Blog Posts

Here are some of my latest blog posts:

{% for post in site.posts %}
  - [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%B %d, %Y" }})
{% endfor %}


## Current Workplace

- Company Name: Fractal Analytics
- Position: AI Engineer
- Brief Summary: 
  - At Fractal Analytics, I'm at the forefront of AI, leading the development of Large Language Models (LLM) applications using retrieval augmented generation (RAG). From customized document generation to architecture development and code conversion, I've got it covered.
  - MLOps is my playground where I streamline pipelines, contribute to data warehouse migration, and develop custom scripts to maintain and monitor the ML models in production.
