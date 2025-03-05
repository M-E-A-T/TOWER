import mido
outputs = mido.get_output_names()
out = mido.open_output(outputs[0])
print(f"output port: {out}")

dict = {}
q = 1

while q:
	track = input("Toggle track: ")
	match track:
		case 'q':
			print("quitting")
			q = 0
		case _:
			t = int(track)
			if t in dict:
				cc = dict[t]
				if cc == 102:
					dict[t] = 0
				else:
					dict[t] = 102
			else:
				dict[t] = 102
			print(f"track: {t}, cc value: {dict[t]}")
			cc_msg = mido.Message('control_change', channel=0, control=t, value=dict[t])
			out.send(cc_msg)
