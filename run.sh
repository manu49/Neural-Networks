data_dir1=$3
data_dir2=$4
out_dir=$5
question=$1
part=$2
extended=$6
if [[ ${out_dir} == "" && ${question} == "1" ]]; then
python3 do_question1.py $part $data_dir1 $data_dir2
fi
if [[ ${out_dir} == "" && ${question} == "2" ]]; then
python3 do_question2.py $part $data_dir1 $data_dir2
fi
if [[ ${out_dir} == "" && ${question} == "3" ]]; then
python3 do_question3.py $part $data_dir1 $data_dir2
fi
if [[ ${question}_${part} == "1_a" ]]; then
python3 do_question1a.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part} == "1_b" ]]; then
python3 do_question1b.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part} == "1_c" ]]; then
python3 do_question1c.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part} == "1_d" ]]; then
python3 do_question1d.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part} == "1_e" ]]; then
python3 do_question1e.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part} == "1_f" ]]; then
python3 do_question1f.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part}_${data_dir1} == "2_a_1" ]]; then
python3 do_question2a1.py $data_dir2 $out_dir $extended
fi
if [[ ${question}_${part}_${data_dir1} == "2_a_2" ]]; then
python3 do_question2a2.py $data_dir2 $out_dir $extended
fi
if [[ ${question}_${part}_${data_dir1} == "2_b_1" ]]; then
python3 do_question2b1.py $data_dir2 $out_dir $extended
fi
if [[ ${question}_${part}_${data_dir1} == "2_b_2" ]]; then
python3 do_question2b2.py $data_dir2 $out_dir $extended
fi
if [[ ${question}_${part}_${data_dir1} == "2_b_3" ]]; then
python3 do_question2b3.py $data_dir2 $out_dir $extended
fi
if [[ ${question}_${part}_${data_dir1} == "2_b_4" ]]; then
python3 do_question2b4.py $data_dir2 $out_dir $extended
fi
if [[ ${question}_${part} == "3_a" ]]; then
python3 do_question3a.py $data_dir1 $data_dir2 $out_dir
fi
if [[ ${question}_${part} == "3_b" ]]; then
python3 do_question3b.py $data_dir1 $data_dir2 $out_dir
fi



# Same for all the parts