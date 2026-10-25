#!/usr/bin/env python3
# A GTK 3 window for the Linux real-backend smoke in CI (Xvfb, openbox, at-spi): every kind of control the backend
# reads and acts on, each with a visible effect the test reads back through AT-SPI. Never run it on a person's desktop.
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gtk  # noqa: E402

GLib.set_prgname('lr-smoke')
GLib.set_application_name('lr-smoke')

win = Gtk.Window(title='LocalRouter smoke')
win.set_default_size(640, 420)
win.move(40, 40)
win.connect('destroy', Gtk.main_quit)

box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=10)
win.add(box)

search = Gtk.Entry()
search.get_accessible().set_name('Search')
box.pack_start(search, False, False, 0)

secret = Gtk.Entry()
secret.set_visibility(False)
secret.set_text('hunter2')
secret.get_accessible().set_name('Password')
box.pack_start(secret, False, False, 0)

status = Gtk.Label(label='Not pressed yet')
box.pack_start(status, False, False, 0)

press = Gtk.Button(label='Press me')
press.connect('clicked', lambda _b: status.set_text('Pressed'))
box.pack_start(press, False, False, 0)

notebook = Gtk.Notebook()
notebook.append_page(Gtk.Label(label='First page'), Gtk.Label(label='First'))
notebook.append_page(Gtk.Label(label='Second page'), Gtk.Label(label='Second'))
box.pack_start(notebook, False, False, 0)

rows = Gtk.ListBox()
for i in range(60):
    rows.add(Gtk.Label(label='Row %d' % i))
scroller = Gtk.ScrolledWindow()
scroller.set_min_content_height(160)
scroller.add(rows)
box.pack_start(scroller, True, True, 0)

win.show_all()
win.present()
Gtk.main()
