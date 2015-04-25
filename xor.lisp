(ql:quickload :nnexp)
(ql:quickload :kutils)

(defvar *xor-network* (make-network-3 2 3 1))
(defvar *test-cases* '((0.0 0.0) (1.0 0.0) (0.0 1.0) (1.0 1.0)))
(defvar *expected* '(0.0 1.0 1.0 0.0))

(defun run-test ()
  (mapcar (kutils:partial #'nnexp:activate-network *xor-network*) *test-cases*))
