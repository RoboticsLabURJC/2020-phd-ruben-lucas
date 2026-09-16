This is a repository to report a Phd progress.


Each week, a file is created in _posts/ directory.

This file is prefixed by the date before it is created

It contains:

Some metadata about the file:
- title
- excerpt

And then, a small description of the expected goals for the week and a reference to an external links where more details can be found.

Each time a new report is requested you should ask for title, excerpt, expected goals and external link.

to the title, you must append in parenthesis which weeks are contained in this report, which is infered from from last report title and prefix.
Examples of title: 

title: Doing some stuff (October week 3)
title: Doing this and this (April week 1 - week 2)

If no excerpt or title is provided, copy and paste the one in the last report (always leaving the date of this week).

### Automated Google Slides Integration
To automate creating and linking the weekly Google Slides presentation, run these steps using the Python 3.10 utility at `google_workspace/gapi_client.py`:
1. **Create MMDD Folder:** Create a folder for the current post date in MMDD format (e.g., `0916` for Sept 16th) inside the parent folder `1WXzwdW8NDzy3N1xzaJef5WIFXve1nsYE`.
2. **Copy Template:** Copy the default template slides (`1HTHAqs0trMnh6zub46YQCZ2j3sfqzrVWC2u28gpJ-Zc`) into the new folder and name the copy MMDD (e.g., `0916`).
3. **Update Bullet Points:** Replace the three bullet points on Slide 1 (representing last week's progress) with the new expected goals specified in the new markdown post. You can use the `replace` action with a `--json-map` on the new presentation ID.
4. **Update Markdown Link:** Update the markdown post's Slides link to point to your newly created weekly presentation instead of the template.

Finally, ask for permission to run the following commands

git add .
git commit -m "updated this week post"
git push origin master

And run all of them at once
