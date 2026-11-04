; Asmnet Complex Number Library (double precision SSE)
; System V AMD64 ABI
; Complex args: re=xmm0, im=xmm1 (operand1), xmm2/xmm3 (operand2)
; Output pointers: rdi = out_re, rsi = out_im  (integer args come after SSE args)

        global  m_cadd
        global  m_csub
        global  m_cmul
        global  m_cdiv
        global  m_cabs
        global  m_cconj
        global  m_cneg

        section .text

; ------------------------------------------------------------
; void m_cadd(are, aim, bre, bim, double *out_re, double *out_im)
; ------------------------------------------------------------
m_cadd:
        addsd   xmm0, xmm2      ; re = are + bre
        addsd   xmm1, xmm3      ; im = aim + bim
        movsd   [rdi], xmm0
        movsd   [rsi], xmm1
        ret

; ------------------------------------------------------------
; void m_csub(...)
; ------------------------------------------------------------
m_csub:
        subsd   xmm0, xmm2
        subsd   xmm1, xmm3
        movsd   [rdi], xmm0
        movsd   [rsi], xmm1
        ret

; ------------------------------------------------------------
; void m_cmul(...)
;   (a+bi)(c+di) = (ac - bd) + (ad + bc)i
; ------------------------------------------------------------
m_cmul:
        push    rbx
        movsd   xmm4, xmm0      ; a
        movsd   xmm5, xmm1      ; b
        movsd   xmm6, xmm2      ; c
        movsd   xmm7, xmm3      ; d

        mulsd   xmm0, xmm6      ; a*c
        mulsd   xmm1, xmm7      ; b*d
        subsd   xmm0, xmm1      ; re = ac - bd

        mulsd   xmm4, xmm7      ; a*d
        mulsd   xmm5, xmm6      ; b*c
        addsd   xmm4, xmm5      ; im = ad + bc
        movsd   xmm1, xmm4

        movsd   [rdi], xmm0
        movsd   [rsi], xmm1
        pop     rbx
        ret

; ------------------------------------------------------------
; void m_cdiv(...)
;   (a+bi)/(c+di) = ((ac+bd)/(c^2+d^2)) + ((bc-ad)/(c^2+d^2))i
; ------------------------------------------------------------
m_cdiv:
        push    rbx
        movsd   xmm4, xmm0      ; a
        movsd   xmm5, xmm1      ; b
        movsd   xmm6, xmm2      ; c
        movsd   xmm7, xmm3      ; d

        mulsd   xmm0, xmm6      ; a*c
        mulsd   xmm1, xmm7      ; b*d
        addsd   xmm0, xmm1      ; num_re = ac + bd

        mulsd   xmm4, xmm7      ; a*d
        mulsd   xmm5, xmm6      ; b*c
        subsd   xmm5, xmm4      ; num_im = bc - ad

        mulsd   xmm6, xmm6      ; c^2
        mulsd   xmm7, xmm7      ; d^2
        addsd   xmm6, xmm7      ; denom

        divsd   xmm0, xmm6      ; re
        divsd   xmm5, xmm6      ; im
        movsd   xmm1, xmm5

        movsd   [rdi], xmm0
        movsd   [rsi], xmm1
        pop     rbx
        ret

; ------------------------------------------------------------
; double m_cabs(are, aim)  returns xmm0 = sqrt(re^2 + im^2)
;   rdi unused (return via xmm0)
; ------------------------------------------------------------
m_cabs:
        mulsd   xmm0, xmm0
        mulsd   xmm1, xmm1
        addsd   xmm0, xmm1
        sqrtsd  xmm0, xmm0
        ret

; ------------------------------------------------------------
; void m_cconj(are, aim, out_re, out_im)  -> re, -im
; ------------------------------------------------------------
m_cconj:
        movabs  rax, 0x8000000000000000   ; sign bit mask
        movq    xmm2, rax
        xorpd   xmm1, xmm2                ; -im
        movsd   [rdi], xmm0               ; re unchanged
        movsd   [rsi], xmm1
        ret

; ------------------------------------------------------------
; void m_cneg(are, aim, out_re, out_im)  -> -re, -im
; ------------------------------------------------------------
m_cneg:
        pxor    xmm2, xmm2
        subsd   xmm2, xmm0                ; -re
        pxor    xmm3, xmm3
        subsd   xmm3, xmm1                ; -im
        movsd   [rdi], xmm2
        movsd   [rsi], xmm3
        ret
