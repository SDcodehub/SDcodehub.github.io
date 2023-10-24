---
layout: default  # Use your preferred layout, e.g., 'default', 'page', etc.
title: Blog
---

## Recent Blog Posts

Here are some of my latest blog posts:

{% for post in site.posts %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
  <p>{{ post.date | date: "%B %d, %Y" }}</p>
  <!-- Add other content if needed -->
{% endfor %}
