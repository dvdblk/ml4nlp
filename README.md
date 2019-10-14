# ML4NLP Exercises Repository

## Zipping script:

Zips the relevant files from an exercise folder. The `ex0[1-4].zip` file is created iff:
    * Folder's name has the format `^(ex0)[1-4]`
    * `lab_report.pdf` is in the folder
    * at least one `ex0[1-4]_*.py` file is in the folder

### To zip all the exercises:

```
~$ sh zip_exercises.sh
Zipping ex01...
Zipping ex02...
Zipping ex03...
~$
```

To zip a specific exercise nr. (e.g. ex02):

```
~$ sh zip_exercises.sh -n 2
Zipping ex02...
```
