---
layout: default
---

<div style="display: flex; justify-content: center; align-items: center;">
  <div style="flex: 1; text-align: center;">
    <!-- Add your profile photo here -->
    <img src="/assets_files/profile.JPG" alt="Your Name" style="border-radius: 50%; max-width: 150px;">
  </div>
  <div style="flex: 2; text-align: center;">
    <!-- Add your tagline here with emojis and slant -->
    <p>I don't know YET, but I'm skilled at finding the answers.</p>

    <!-- Add your tagline here with emojis and slant -->
    <p>🚀 Learn and Experiment with AI technology 🧠</p>
    
    <!-- Add your current interest here -->
    <p>LLM | Agents | MLOps | ML</p>
    
    <!-- Navigation Links -->
    <div>
      <p></p>
      <ul>
        <a href="https://www.linkedin.com/in/sagar-s-desai/">LinkedIN</a> |
        <a href="https://github.com/SDcodehub">GitHub</a> |
        <a href="https://huggingface.co/sagarsdesai">Hugging Face</a> |
        <a href="https://www.kaggle.com/sagarsdesai" target="_blank">Kaggle</a> 
      </ul>
    </div>
  </div>
</div>

<br> <!-- Add an empty line here -->

---

## Pet Projects

{% for project in site.pages %}
  {% if project.project_type == 'project' %}
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
