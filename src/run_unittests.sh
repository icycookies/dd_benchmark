for file in $(ls ./unittests/)
do
    if [ file != "__init__.py" ]; then
        python3 ./unittests/$file
    fi
done