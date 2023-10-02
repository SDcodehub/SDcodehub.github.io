---
layout: default
title: Projects
permalink: /projects/
---

# Projects

{% for project in site.projects %}
  <h2><a href="{{ project.url }}">{{ project.title }}</a></h2>
  <img src="{{ project.image }}" alt="{{ project.title }}">
  <p>{{ project.description }}</p>
{% endfor %}
