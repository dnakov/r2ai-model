#!/bin/sh

F="$1"
[ -z "$EDITOR" ] && EDITOR=vim

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
	echo "> (i)gnore (o)k (e)dit (r)emove (q)uit"
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
	r)
		rm_lastline
		;;
	e)
		cat_lastline > "$F".edit
		$EDITOR "$F".edit
		if [ -s "$F".edit ]; then
			rm_lastline
			head -n1 "$F".edit >> "$F"
		fi
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
