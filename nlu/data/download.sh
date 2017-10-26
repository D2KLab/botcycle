source .env


# get the token (faster in python)
WIT_DOWNLOAD_EN=`python get_link.py $WIT_TOKEN_EN BotCycle_en`
WIT_URL_EN="https://api.wit.ai/export/$WIT_DOWNLOAD_EN"

rm -rf wit_en
wget -O wit_data_en.zip $WIT_URL_EN
unzip -o wit_data_en.zip -d wit_en
rm -rf wit_data_en.zip
mv wit_en/BotCycle_en wit_en/source

WIT_DOWNLOAD_IT=`python get_link.py $WIT_TOKEN_IT BotCycle_it`
WIT_URL_IT="https://api.wit.ai/export/$WIT_DOWNLOAD_IT"

rm -rf wit_it
wget -O wit_data_it.zip $WIT_URL_IT
unzip -o wit_data_it.zip -d wit_it
rm -rf wit_data_it.zip
mv wit_it/BotCycle_it wit_it/source
