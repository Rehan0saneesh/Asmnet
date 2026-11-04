read -p "File: " filename

nasm -f elf64 -o "$filename".o "$filename".asm
ld -o program "$filename".o
./program


rm "$filename".o
rm program
