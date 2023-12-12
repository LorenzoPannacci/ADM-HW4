#!/bin/bash

# /!\ ATTENTION /!\

# Into the Q2 we need the 'mktime' function which is not presente natively in the awk command of macOS
# So if you are using a Mac you need to install the gawk command (GNU Awk) manually which is an extended and more powerful version of awk

# We can do this with a packages manager for macOS like Homebrew
# To install Homebrew we can pasting and executing the following command into the terminal

# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# This command can also been found on this link --> https://brew.sh/

# After installing Homebrew we can install gawk by pasting and executing the following command in the terminal
# brew install gawk

echo -e "\n"
echo "ADM-HW4 Winter Semester 2023"
echo -e "\n"
echo "# CLQ - Group 19"
echo -e "\n"
echo "## Answer the following questions:"
echo -e "\n"
echo "Q1) What is the most-watched Neflix title?  "


# The 'awk' command allows to execute an action specified between braces to all the lines of a file that respect a certain pattern ---> example:  awk 'pattern { action }' file
# -F'\t'  --> Specifies the field separator as a tab character  --> this is the field separator of a .csv file
# gsub(/ /, "_", $4)  -->  Replaces spaces with underscores in the 4th column 
#                           --> This step is essential to ensure that occurrences are sorted correctly since the presence of empty spaces creates problems in sorting
# count[$4]++         -->  Increments the count for each unique value in the 4th column
# END {for (country in count) print country, count[country]} --> After the action specified between the braces it iterates over the array of counts and prints each title with its count
# sort -k2,2nr  --> Sorts based on the second column (count of the title = 4th column) in reverse numerical order
# head -n1      --> Output only the first line of the sorted couple of title and count of the title
# read -r country count1 <<< --> Read the output assigned to the two variables 

read -r answer_q1 count_q1 <<< "$(awk -F',' '{ gsub(/ /, "_", $4); count[$4]++ } END {for (title in count) print title, count[title]}' vodclickstream_uk_movies_03.csv | sort -k2,2nr | head -n1)"

# replace the _ with a space into the name of the most watched title 
answer_q1="${answer_q1//_/ }"

# N.B. we thought that was more correct to chose the title that has been watched the most number of times ('title' column)
#      instead the title that was watched for the most time ('duration' column) since it would have been influenced by the length of the title
echo "- The most watched title on Netflix is '$answer_q1' witch was watched $count_q1 times"

#---------------------------------------------------------------------------------------------------------

echo -e "\n"
echo "Q2) Report the average time between subsequnet clicks on Netfix"


# Sort the dataset based on the second column
sorted_file=$(sort -t',' -k2,2 vodclickstream_uk_movies_03.csv)

# The gawk is a more powerful version of awk 
# since we haven't a pattern we will execute the action between the brace to all the rows of the sorted dataset
gawk -F, '{
    # Convert the 2nd column into a timestamp epoch format with the function mktimestamp which requires a date in the format YYYY MM DD HH MM SS
    # the timestamp epoch is the number of seconds that are elapsed from Jan 1st 1970 to the current date
    # gensub(/[-:]/, " ", "g", $2) sobstitute all the - and : with a space into the 2nd column
    #                              --> the paramether g specifies that we will apply the substitution to all the matchs and not only to the first

    current_row = mktime(gensub(/[-:]/, " ", "g", $2));

    # Calculate the difference in seconds from the previous row
    # the if condition exclude the calculus of the difference if we are in the first row
    # the row number is obtained using the built-in variable 'NR' which keep the count of the number of input records read so far

    if (NR > 1) {
        difference = current_row - prev_row;
        cumulative_sum += difference;

        # Print the number of the row, the datetime value, the differences with the previus row in seconds and the cumulative sum
        # use this print only if you want to see the progress 
        # print NR "  |  " $2 "  |  " difference "  |  " cumulative_sum;
    }

    # Save the current rows for the next iteration 
    prev_row = current_row;
} END {
    # Print at the end of the end of ... the cumulative sum 
    # the if condition prevent to print the cumulative sum if the dataset has only 1 row
    
    if (NR > 1) {

        # Remove the last difference because it is a negative number which should not be considered
        # --> the actual number of difference is the number of rows - 1

        cumulative_sum= (cumulative_sum-difference)
        average_diff = cumulative_sum / (NR - 1)

        print "- The average time between two subsequent clicks on Netflix is: " average_diff " seconds";
    }
}' <<< "$sorted_file"


#---------------------------------------------------------------------------------------------------------

echo -e "\n"
echo "Q3) Provide the ID of the user that has spent the most time on Netflix"


# The 'awk' command allows to execute an action specified between braces to all the lines of a file that respect a certain pattern ---> example:  awk 'pattern { action }' file
# -F'\t'  --> Specifies the field separator as a tab character  --> this is the field separator of a .csv file
# {duration[$NF] += $3}  ---> grupify the rows for the last column selecteb with the built-in variable 'NF' which contain the number of fields of the dataset
#                             ---> then we summ the values in the 3rd column for each unique value in the last column
# END { for (user_id in duration) print user_id, duration[user_id] }  --> After the action specified between the braces it iterates over the array of unique user_id and prints each user_id with its cumulative duration
# sort -k2,2nr  --> Sorts based on the second column (cumulative duration of the user_id = last column) in reverse numerical order
# head -n1      --> Output only the first line of the sorted couple of user_id and cumulative duration 
# read -r country count1 <<< --> Read the output assigned to the two variables 

read -r ID time_spent <<< "$(awk -F',' '{ duration[$NF] += $3 } END { for (user_id in duration) print user_id, duration[user_id] }' vodclickstream_uk_movies_03.csv | sort -k2,2nr | head -n1)" 

echo "- The user that has spent the most time on Netflix is $ID which has spent $time_spent seconds on Netlix"
echo -e "\n"
