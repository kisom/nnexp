;;;; nnexp.asd

(asdf:defsystem #:nnexp
  :description "Experiments in neural networks."
  :author "Kyle Isom <kyle@metacircular.net>"
  :license "MIT License"
  :serial t
  :depends-on (#:l-math)
  :components ((:file "package")
               (:file "nnexp")))

