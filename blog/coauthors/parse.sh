IN=raw_export.bib
OUT=hal_biblio.bib

cp $IN $OUT
sed -i -e "s/{\\\'a}/a/g" \
    -e "s/{\\\'e}/e/g" \
    -e "s/{\\\'i}/i/g" \
    -e "s/{\\\'n}/n/g" \
    -e "s/{\\\'o}/o/g" \
    -e "s/{\\\'u}/u/g" \
    -e "s/{\\\'y}/y/g" \
    -e "s/{\\\'c}/c/g" \
    -e "s/{\\\'A}/A/g" \
    -e "s/{\\\'E}/E/g" \
    -e 's/{\\`a}/a/g' \
    -e 's/{\\`e}/e/g' \
    -e 's/{\\`i}/i/g' \
    -e 's/{\\`o}/o/g' \
    -e 's/{\\`u}/u/g' \
    -e 's/{\\`A}/A/g' \
    -e 's/{\\`E}/E/g' \
    -e 's/{\\^a}/a/g' \
    -e 's/{\\^e}/e/g' \
    -e 's/{\\^i}/i/g' \
    -e 's/{\\^o}/o/g' \
    -e 's/{\\^u}/u/g' \
    -e 's/{\\c c}/c/g' \
    -e 's/{\\c s}/s/g' \
    -e 's/{\\v c}/c/g' \
    -e 's/{\\v e}/e/g' \
    -e 's/{\\v s}/s/g' \
    -e 's/{\\v z}/z/g' \
    -e 's/{\\v S}/S/g' \
    -e 's/{\\u a}/a/g' \
    -e "s/D '/D'/g" \
    -e 's/{\\"a}/a/g' \
    -e 's/{\\"e}/e/g' \
    -e 's/{\\"i}/i/g' \
    -e 's/{\\"o}/o/g' \
    -e 's/{\\"u}/u/g' \
    -e 's/{\\"O}/O/g' \
    -e 's/{\\~n}/n/g' \
    -e 's/{\\~a}/a/g' \
    -e 's/{\\i}/i/g' \
    -e 's/{\\o}/o/g' \
    -e 's/{{\\O}}/O/g' \
    -e 's/{o}/o/g' \
    -e 's/{\\DH}/Dh/g' \
    -e 's/{\\dh}/dh/g' \
    -e 's/{\\dj}/dj/g' \
    -e 's/{\\aa}/aa/g' \
    -e 's/{\\oe}/oe/g' \
    -e 's/{\\ae}/ae/g' \
    -e 's/{\\ss}/ss/g' \
    $OUT
