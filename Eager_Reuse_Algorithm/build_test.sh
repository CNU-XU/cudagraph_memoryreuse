rm -rf result
mkdir result
if [ $# == 0 ]; then
	loop=1
else
	loop=$1
fi
gcc -o total_test total_test.c
i=0
while [ $loop -gt $i ]
do
	echo ========test$i========
	./total_test result_$i
	grep memory_size result/result_$i
	i=$[$i+1]
done
rm -rf total_test
