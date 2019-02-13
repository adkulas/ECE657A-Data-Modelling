# ECE657A-Data-Modelling
Homework Assignments and projects from ECE657A Data and Knowledge Modelling

source: https://gist.githubusercontent.com/jnaecker/da8c1846bc414594783978b66b6e8c83/raw/08da9fba02dc9178fb41775da4481d6e2fff6e64/git+overleaf+github.md
# git + overleaf + github

## Setup

### Connect Overleaf and your local repo

1. Make a new project on [Overleaf](https://www.overleaf.com).
2. In the share menu, copy the link from "Clone with git"
3. On your computer:
    - use `cd` to navigate to where you want to put your project
    - type `mkdir` and then the name of the project
    - `cd` into that project
    - do `git init`
    - do `git remote add overleaf` with the link you copied
    - so overall this would like something like this

```
cd ~/Desktop
mkdir overleaf-project
cd overleaf-project
git init
git remote add overleaf https://git.overleaf.com/11205025wxdxfsqpxytc
git pull overleaf master
```

### Connect your local repo and Github

1. Make a new repo on Github
2. Copy the git remote link
3. On your computer:

```
git remote add github https://github.com/jnaecker/overleaf-project.git
git push github master
```

## Regular Workflow

1. Make some changes on Overleaf
2. On your computer (while in the local repo directory), run

```
git pull overleaf master
git push github master
```

Any changes you made on overleaf should now be on both your local repo and on Github!

If you want to make changes via your local repo, just push to both remote repos (after staging and committing locally):

```
git add .
git cm "Adding stuff from local repo"
git push overleaf master
git push github master
```

You may also want
