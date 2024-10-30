(function() {
 	function r2ai(q) {
		const host = "http://127.0.0.1:8080/cmd/";
		const ss = q.replace(/ /g, "%20");
		r2.syscmd('curl -s ' + host + '/' + "-R > /dev/null");
		const cmd = 'curl -s ' + host + '/' + ss;
		return r2.syscmds(cmd).split(/\n/g)[0].trim();
	}
	const a0 = +r2.cmd("?vi $S");
	const a1 = +r2.cmd("?vi $S+$SS");
	let count = 50;
	for (let a = a0; a < a1; ) {
		const op = r2.cmdj("aoj@ "+a)[0];
		const json = JSON.stringify({description:op.description, opcode: op.opcode, esil:op.esil})
		const q = `tell in one sentence '${op.pseudo}', do not use the expression or introduce your answer. consider this metadata ${json}`
		const ops = [
			op.opcode,
			op.esil,
			op.pseudo,
			r2ai(q),
		];
		console.log(ops.join("\t"));
		a += op.size;
		if (count-- <0) {
			break;
		}
	}
})();
