echo "install requirements"
pip install sphinx sphinx-autobuild

echo "build docs"

mkdir -p builds/
sphinx-build -b html ./docs/ builds/ -W --keep-going

echo "COMMENT=$(git log -1 --pretty=%B)"
COMMENT=$(git log -1 --pretty=%B)

echo "cp -a builds $TEMP/builds"
cp -a builds $TEMP/builds

echo "cd $TEMP"
cd $TEMP

echo "git clone --single-branch --branch gh-pages https://GH_TOKEN:$GH_TOKEN@github.com/catalyst-team/mlcomp.git"
git clone --single-branch --branch gh-pages https://GH_TOKEN:$GH_TOKEN@github.com/catalyst-team/mlcomp.git

echo "copying files"
cd mlcomp
rm -rf *
cp -a $TEMP/builds/* .

echo "git commit and push"
git config --global user.email "teamcity@catalyst.github"
git config --global user.name "Teamcity"
git add .
git commit -m "$COMMENT"
git push