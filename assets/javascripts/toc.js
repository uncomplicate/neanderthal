jQuery.fn.toc = function () {
  if(this.length === 0)
    return;

  var listStack = [ $("<ul class='nav nav-list' />")];
  listStack[0].appendTo(this);

  Array.prototype.last = function() { return this[this.length - 1]};

  var level = 2;
  $(document).ready(function() {
    $(":header").filter(function(idx, el) {
      // filter out H1
      return !(parseInt(el.tagName[1]) === 1);
    }).
    each(function(index, el) {
      var currentLevel = parseInt(el.tagName[1]);

      var text = $(el).text();
      var anchor = text.replace(/[^a-zA-Z 0-9]+/g,'').replace(/\s/g, "_").toLowerCase();

      $(el).attr('id', anchor);

      if(currentLevel > level) {
        var nextLevelList = $("<ul class='nav nav-list'/>");
        nextLevelList.appendTo(listStack.last().children("li").last());
        listStack.push(nextLevelList);
      } else if(currentLevel < level) {
	var delta = level - currentLevel;
        for(var i = 0; i < delta; i ++) {
	  listStack.pop();
	}
      }

      level = currentLevel;
      var li = $("<li />");

      $("<a />").text(text).attr('href', "#" + anchor).appendTo(li);
      li.appendTo(listStack.last());
    });
  });
};

$(document).ready(function() {
  $(".well.sidebar-nav").toc();
});
