#!/bin/sh

F="$1"

if [ -z "$F" ]; then
	echo "Usage: review-pending.sh [tsvfile]"
	exit 1
fi

cat_lastline() {
	tail -n1 "$F"
}

rm_lastline() {
	sed '$d' "$F" > "$F".tmp
	mv "$F".tmp "$F"
}


while : ; do
	echo "============================================"
	echo
	cat_lastline | sed -e 's/\t/\n\n/g'
	echo
	echo "============================================"
	echo "> (i)gnore (o)k (e)dit (q)uit"
	read O
	case "$O" in
	i)
		cat_lastline >> "$F".ignored
		rm_lastline
		;;
	o)
		cat_lastline >> "$F".ok
		rm_lastline
		;;
	e)
		cat_lastline > "$F".edit
		$EDITOR "$F".edit
		rm_lastline
		head -n1 "$F".edit >> "$F"
		rm -f "$F".edit
		;;
	q)
		exit 0
		;;
	*)
		echo "Unknown action"
		;;
	esac
done
