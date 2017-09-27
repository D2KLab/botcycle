
# get the token (faster in python)
WIT_TOKEN=`python get_link.py`
WIT_URL="https://api.wit.ai/export/$WIT_TOKEN"

wget -O wit_data.zip $WIT_URL
rm -rf wit
unzip -o wit_data.zip -d wit
rm -rf wit_data.zip
