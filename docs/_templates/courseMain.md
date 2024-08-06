---
Author: Vighnesh Nayak
Date: <% tp.file.creation_date("DD/MM/YYYY") %>
Course: <% tp.file.folder() %>
tags:
  - hide
  - main
---
# <% tp.file.title %>
---

```dataview
table Author, Date, dateformat(file.mtime,"dd/MM/yyyy") as Modified
from "<% tp.file.folder(true) %>" where file.name!="Main" sort Date desc 
```
