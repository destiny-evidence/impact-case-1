#set page(paper: "presentation-16-9", margin: (x: 2cm, y: 1.6cm), numbering: "1")
#set text(size: 20pt, font: "Liberation Sans")
#set par(leading: 0.6em)
#set figure(supplement: none)
// Figures are fitted into the room below a slide heading, keeping their aspect ratio.
#show image: set image(height: 8.8cm, fit: "contain")

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(1fr)
  text(size: 36pt, it.body)
  v(1fr)
}
#show heading.where(level: 2): it => {
  pagebreak(weak: true)
  text(size: 28pt, it.body)
  v(0.4em)
}

#align(center + horizon)[
  #text(size: 40pt, weight: "bold")[[-doc.title-]]
  [# if doc.authors #]

  [# for author in doc.authors #][- author.name -][# if not loop.last #], [# endif #][# endfor #]
  [# endif #]
]

[-IMPORTS-]

// MyST tables use tablex with these (empty by default) style dictionaries.
#let tableStyle = (
  stroke: 0.4pt + luma(170),
  inset: 0.7em,
  fill: (col, row) => if row == 0 { luma(235) },
)
#show table: set text(size: 18pt)

[-CONTENT-]
