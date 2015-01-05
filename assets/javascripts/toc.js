$(document).ready(function() {
  var $toc = $('#toc');

  function format (item) {
    if (item.children && item.children.length > 0) {
      return "<li> <a href=#" + item.id + ">" + item.title + "</a><ul class='nav'>"
        + item.children.map(format).join('')
        + "</ul></li>";
    } else {
      return "<li> <a href=#" + item.id + ">" + item.title + "</a></li>";
    }
  }
  // return;

  if($toc.length) {
    var $h3s = $('.container .col-md-9 :header');

    var tocTree = [];
    var lastRoot;

    $h3s.each(function(i, el) {
      var $el = $(el);
      var id = $el.attr('id');
      var title = $el.text();
      var depth = parseInt($el.prop("tagName")[1]);

      if(depth > 3)
        return;

      if (lastRoot && depth > lastRoot.depth) {
        lastRoot.children.push({id: id, title: title });
      } else {
        lastRoot = {depth: depth,
                    title: title,
                    id: id,
                    children: []};
        tocTree.push(lastRoot);
      }
    });

    var titles = tocTree.map(format).join('');

    $toc.html(titles);
  }

  $("#toc").parent().affix();

  $('#side-navigation').on('activate.bs.scrollspy', function (e) {
    var parent = $(e.target).parent().parent()[0];
    if (parent.tagName == "LI") {
      $(parent).addClass("active");
    }

  });

});
