#!/bin/bash

echo -e "\n"
echo "ADM-HW4 Winter Semester 2023"
echo -e "\n"
echo "# CLQ - Group 19"
echo -e "\n"
echo "## Load the Dataset vodclickstream_uk_movies_03.csv"

#cat vodclickstream_uk_movies_03.csv | head -n 10
#echo "Done ✅"

echo -e "\n"
echo "## Answer the following questions:"
echo -e "\n"
echo "Q1) What is the most-watched Neflix title?  "

read -r answer_q1 count_q1 <<< "$(awk -F',' 'NR>1 { gsub(/ /, "_", $4); count[$4]++ } END {for (title in count) print title, count[title]}' vodclickstream_uk_movies_03.csv | sort -k2,2nr | head -n1)"

# replace the _ with a space 
answer_q1="${answer_q1//_/ }"

# N.B. i think that is more correct chose the title that has been watched the most number of times ('title' column)
#      instead the title that was watched for the most time ('duration' column)
echo "- The most watched title on Netflix is '$answer_q1' witch was watched $count_q1 times"

#---------------------------------------------------------------------------------------------------------

echo -e "\n"
echo "Q2) Report the average time between subsequnet clicks on Netfix"


input_file="vodclickstream_uk_movies_03.csv"

# use this to extract a partition of the df 
#head -n 10 "$input_file" > temp_file.csv

n_max=$(wc -l < vodclickstream_uk_movies_03.csv)

echo $n_max

# Convert the 2nd colonn into a datetime formato
awk -F',' 'BEGIN {OFS=FS} NR==1 {print; next} {gsub(/ /,"T",$2); print}' vodclickstream_uk_movies_03.csv\
    | sort -t',' -k2,2 \
    | tee sorted_file.csv \
    | awk -F',' '{print $2}' \
    | {
        read -r current_row
        current_timestamp=$(date -jf "%Y-%m-%dT%H:%M:%S" "$current_row" +"%s")

        sum_diff=0
        num_rows=2

        while [ $num_rows -lt $n_max ]; do
            read -r next_row
            next_timestamp=$(date -jf "%Y-%m-%dT%H:%M:%S" "$next_row" +"%s")
            diff=$((next_timestamp - current_timestamp))
            sum_diff=$((sum_diff + diff))
            current_timestamp=$next_timestamp
            echo "riga: $num_rows | differenza cumulata: $sum_diff  | differenza: $diff"
            ((num_rows++)) 
        done
        ((num_rows--))
        average_diff=$((sum_diff / num_rows))
        echo "Cumulative difference: $sum_diff | total number o difference: $num_rows | average difference: $average_diff seconds"
    }


#---------------------------------------------------------------------------------------------------------

echo -e "\n"
echo "Q3) Provide the ID of the user that has spent the most time on Netflix"

read -r ID time_spent <<< "$(awk -F',' 'NR>1 { duration[$NF] += $3 } END { for (user_id in duration) print user_id, duration[user_id] }' vodclickstream_uk_movies_03.csv | sort -k2,2nr | head -n1)" 

echo "- The user that has spent the most time on Netflix is $ID which has spent $time_spent seconds on Netlix"

