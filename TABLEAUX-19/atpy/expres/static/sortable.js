
function isInt(n)
{
   return Number(n) === n && n % 1 === 0;
}

function isFloat(n)
{
   return Number(n) === n && n % 1 !== 0;
}

function createLegend(data, id)
{
   var legend = document.createElement('table');

   if (!("LEGEND" in data))
   {
      return legend;
   }

   for (var i in data["LEGEND"]) 
   {
      var tr = document.createElement('tr');
      var th = document.createElement('th');
      var td = document.createElement('td');

      th.appendChild(document.createTextNode(i));
      td.appendChild(document.createTextNode(data["LEGEND"][i]));

      tr.appendChild(th);
      tr.appendChild(td);

      legend.appendChild(tr);
   }

   return legend;
}


function createTable(table_data, id, sortIndex, rev) 
{
   data = table_data["DATA"];
   header = table_data["HEADER"];
   data.sort(function(rowx, rowy) {
      var x = rowx[sortIndex];
      var y = rowy[sortIndex];
      return ((x<y) ? -1 : ((x>y) ? 1 : 0)) * rev;
   });
   
   var table = document.createElement('table');
   table.setAttribute('id', id);
   
   // header
   var tr = document.createElement('tr');
   for (var i=0; i<header.length; i++) 
   {
      var th = document.createElement('th');
      th.setAttribute('onclick', onclickSort(id, i, rev));
      th.appendChild(document.createTextNode(header[i]));
      tr.appendChild(th);
   }
   table.appendChild(tr);

   // rows
   for (var j=0; j<data.length; j++)
   {
      var name = data[j][0];
      var tr = document.createElement('tr');
      if (name in table_data["CLASSES"]) 
      {
         tr.classList.add(table_data["CLASSES"][name]);
      }
      if (name in highlighted) 
      {
         tr.classList.add(highlighted[name]);
      }
      for (var i=0; i<header.length; i++) 
      {
         var td = document.createElement('td');
         if (i == 0) {
            //td.setAttribute("class", "name "+name);
            td.classList.add("name", name);
            td.setAttribute("onclick", onclickHighlight(name));
         }
         var val = data[j][i];
         if (isFloat(val)) {
            val = val.toFixed(1);
         }
         td.appendChild(document.createTextNode(val));
         tr.appendChild(td);
      }
      table.appendChild(tr);
   }

   return table;
}

function updateLegend(data, id)
{
   var oldLegend = document.getElementById(id);
   var newLegend = createLegend(data, id);
   oldLegend.parentNode.replaceChild(newLegend, oldLegend);
}

function updateTable(table, id, sortIndex, rev)
{
   var oldTable = document.getElementById(id);
   var newTable = createTable(table, id, sortIndex, rev);
   oldTable.parentNode.replaceChild(newTable, oldTable);
}

function onclickSort(id, index, rev)
{
   return 'updateTable('+id+',"'+id+'",'+index+','+rev+')';
}

function onclickHighlight(name)
{
   return 'highlight("'+name+'")';
}

var highlighted = {};
function highlight(name)
{
   if (name in highlighted) {
      delete highlighted[name];
   }
   else 
   {
      highlighted[name] = "sel";
   }
   
   var elements = document.getElementsByClassName(name);
   for (var i = 0; i < elements.length; i++) 
   {  
      if (name in highlighted) {
         elements[i].parentNode.classList.add("sel");
      }
      else 
      {
         elements[i].parentNode.classList.remove("sel");
      }
   }
}

