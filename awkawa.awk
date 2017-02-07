BEGIN{
environment="cat dog thunder phonering alarm water steps bird;"
voice="femalemale";
music="funksoulrnb rock jazz raphiphop";
}
/^Class:/ {class=$2;next}
{
	/^Results:/ {print $1}
#{if (environment ~ class)
#	{print class
#	}
#}
