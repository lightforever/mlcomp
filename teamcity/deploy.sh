pip install sphinx sphinx-autobuild

mkdir -p builds/
sphinx-build -b html ./docs/ builds/ -W --keep-going

COMMENT=$(git log -1 --pretty=%B)

cp -a builds $TEMP/builds

cd $TEMP

git clone --single-branch --branch gh-pages https://GH_TOKEN:$GH_TOKEN@github.com/catalyst-team/mlcomp.git

cd mlcomp
rm -rf *
cp -a $TEMP/builds/* .

if [ $GIT_BRANCH == 'refs/heads/master' ]; then
  git config --global user.email "teamcity@catalyst.github"
  git config --global user.name "Teamcity"
  git add .
  git commit -m "$COMMENT"
  git push
fi