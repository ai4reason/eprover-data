import os
from . import details, summary, jsdata

HTML_DIR = os.getenv("EXPRES_HTML", 
   os.path.join(os.getenv("HOME"),"public_html","expres"))

BEGIN = '<html>\n<head>\n'
TITLE = '\t<title>%s</title>\n'
CSS = '\t<link rel="stylesheet" type="text/css" href="../static/style.css">\n'
JSLIB = '\t<script src="../static/sortable.js"></script>\n'
JSDATA = '\t<script src="data/%s.js"></script>\n'
ONLOAD = '\t<script>window.onload = function() { %s };</script>\n'
BODY = '</head>\n<body>\n\n'
H1 = '<h1>%s</h1>\n'
H2 = '<h2>%s</h2>\n'
TABLE = '<div class="tables"><div class="box"><table id="%s"></table></div></div>\n'
END = '</body>\n</html>\n'

def path(f_name):
   return os.path.join(HTML_DIR, f_name)

def onload(data, h_table, h_legend):
   js = " "
   if h_table:
      js += 'updateTable(%s, "%s", 3, -1); ' % (data, data)
   if h_legend:
      js += 'updateLegend(%s, "legend___%s"); ' % (data, data)
   return js

def begin(out, title, data, exp, h_table, h_legend, ref=None):
   out.write(BEGIN)
   out.write(TITLE % title)
   out.write(CSS)
   out.write(JSLIB)
   out.write(JSDATA % data)
   out.write(ONLOAD % onload(data, h_table, h_legend))
   out.write(BODY)
   out.write(H1 % title)
   out.write('<p>experiment id: %s\n' % exp)
   out.write('<br>data id: %s\n\n' % data)
   if ref:
      out.write('<br>ref: %s\n\n' % ref)

def legend(out, data):
   out.write(H2 % "Legend")
   out.write(TABLE % ("legend___%s" % data))
   out.write("\n")

def table(out, data, title="Results"):
   out.write(H2 % title)
   out.write(TABLE % data)
   out.write("\n")

def end(out):
   out.write(END)
   out.close()

def create(exp, data):
   f_out = path(os.path.join(exp, data+".html"))
   os.system("mkdir -p %s" % os.path.dirname(f_out))
   out = file(f_out, "w")
   return out

def processed(bid, pids, results, exp="results", data="data"): 
   proc = details.processed(bid, pids, results)
  
   out = create(exp, data)
   begin(out, "Processed @ %s" % bid, data, exp, h_table=True, h_legend=True)
   legend(out, data)
   table(out, data)
   end(out)

   f_js = path(os.path.join(exp, "data", data+".js"))
   leg = dict(enumerate(pids))
   header = ["problem"]+leg.keys()
   rows = [[d[0]]+[proc[d][pid] for pid in pids] for d in sorted(proc)]
   jsdata.save(f_js, data, header, {}, rows, leg)

def solved(bid, pids, limit, results, exp="results", ref_pid=None, multi_pid=True):
   stat = summary.make(bid, pids, results, ref=ref_pid)
   if isinstance(limit, int):
      limit = "%ss" % limit
   else:
      limit = limit.replace("+","")

   if (not multi_pid) and ref_pid:
      data = ("summary---%s---%s---%s" % (bid.replace("/","_"),ref_pid,limit)).replace("-","_")
   else:
      data = ("summary---%s---%s" % (bid.replace("/","_"),limit)).replace("-","_")
   out = create(exp, data)
   begin(out, "Summary @ %s @ %s" % (bid, limit), data, exp, h_table=True, h_legend=False, ref=ref_pid)
   table(out, data)
   end(out)
   
   f_js = path(os.path.join(exp, "data", data+".js"))
   rows = [[pid]+stat[pid] for pid in stat]
   if os.path.isfile(f_js):
      jsdata.update(f_js, data, rows, key=lambda row: row[0])
   else:
      header = ["proto", "total", "errors", "solved"]
      classes = {}
      if ref_pid: 
         header += ["plus", "minus"]
         classes[ref_pid] = "ref"
      jsdata.save(f_js, data, header, classes, rows, None)

