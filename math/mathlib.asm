; Asmnet Math Library
; Follows System V AMD64 ABI:
;   int args : rdi, rsi, rdx, rcx, r8, r9
;   float args: xmm0..xmm7
;   return   : rax (int) / xmm0 (float)
; Callee-saved registers preserved: rbx, rbp, r12-r15

        global  m_add
        global  m_sub
        global  m_mul
        global  m_div
        global  m_mod
        global  m_neg

        global  m_pow_i
        global  m_sqrt
        global  m_factorial
        global  m_gcd
        global  m_lcm

        global  m_fadd
        global  m_fsub
        global  m_fmul
        global  m_fdiv
        global  m_fneg
        global  m_fabs

        global  m_min
        global  m_max
        global  m_abs

        ; scalar float transcendental / misc
        global  f_sqrt
        global  f_sin
        global  f_cos
        global  f_exp
        global  f_log
        global  f_sigmoid
        global  f_erf
        global  f_square
        global  i_square
        global  f_threshold

        section .rodata
e_const:    dq 2.718281828459045
ln2_const:  dq 0.6931471805599453
one_const:  dq 1.0
zero_const: dq 0.0
p_const:    dq 0.3275911
a1_const:   dq 0.254829592
a2_const:   dq -0.284496736
a3_const:   dq 1.421413741
a4_const:   dq -1.453152027
a5_const:   dq 1.061405429

        global  e_const
        global  ln2_const
        global  one_const
        global  zero_const
        global  p_const
        global  a1_const
        global  a2_const
        global  a3_const
        global  a4_const
        global  a5_const

        section .text

; ------------------------------------------------------------
; Integer arithmetic
; ------------------------------------------------------------
m_add:          ; long m_add(long a, long b)
        mov     rax, rdi
        add     rax, rsi
        ret

m_sub:          ; long m_sub(long a, long b)
        mov     rax, rdi
        sub     rax, rsi
        ret

m_mul:          ; long m_mul(long a, long b)
        mov     rax, rdi
        imul    rax, rsi
        ret

m_div:          ; long m_div(long a, long b)
        mov     rax, rdi
        cqo
        idiv    rsi
        ret                     ; quotient in rax

m_mod:          ; long m_mod(long a, long b)
        mov     rax, rdi
        cqo
        idiv    rsi
        mov     rax, rdx        ; remainder
        ret

m_neg:          ; long m_neg(long a)
        mov     rax, rdi
        neg     rax
        ret

; ------------------------------------------------------------
; Integer power: m_pow_i(base, exp)  (exp >= 0)
; ------------------------------------------------------------
m_pow_i:
        push    rbx
        mov     rax, 1
        mov     rbx, rdi        ; base
        cmp     rsi, 0
        jle     .done
.loop:
        test    rsi, rsi
        jz      .done
        test    rsi, 1
        jz      .skip_mul
        imul    rax, rbx
.skip_mul:
        imul    rbx, rbx
        shr     rsi, 1
        jmp     .loop
.done:
        pop     rbx
        ret

; ------------------------------------------------------------
; Integer square root (floor): m_sqrt(n)
;   largest x such that x*x <= n
; ------------------------------------------------------------
m_sqrt:
        push    rbx
        push    rcx
        mov     rax, rdi
        xor     rbx, rbx        ; result
        xor     rcx, rcx        ; candidate
        cmp     rax, 0
        jle     .done
.search:
        inc     rcx
        mov     rdx, rcx
        imul    rdx, rdx
        cmp     rdx, rax
        jbe     .set
        dec     rcx
        jmp     .done
.set:
        mov     rbx, rcx
        jmp     .search
.done:
        mov     rax, rbx
        pop     rcx
        pop     rbx
        ret

; ------------------------------------------------------------
; Factorial: m_factorial(n)  (n >= 0, small)
; ------------------------------------------------------------
m_factorial:
        push    rbx
        mov     rax, 1
        mov     rbx, rdi
        cmp     rdi, 1
        jle     .done
.loop:
        imul    rax, rbx
        dec     rbx
        cmp     rbx, 1
        jg      .loop
.done:
        pop     rbx
        ret

; ------------------------------------------------------------
; GCD (Euclidean): m_gcd(a, b)
; ------------------------------------------------------------
m_gcd:
        push    rbx
        mov     rax, rdi
        mov     rbx, rsi
        test    rbx, rbx
        jz      .done
.loop:
        xor     rdx, rdx
        div     rbx
        mov     rax, rbx
        mov     rbx, rdx
        test    rdx, rdx
        jnz     .loop
.done:
        pop     rbx
        ret

; ------------------------------------------------------------
; LCM: m_lcm(a, b) = a / gcd(a,b) * b
; ------------------------------------------------------------
m_lcm:
        push    rbx
        mov     rdx, rdi
        call    m_gcd
        mov     rbx, rax        ; gcd
        mov     rax, rdi
        xor     rdx, rdx
        idiv    rbx
        imul    rax, rsi
        pop     rbx
        ret

; ------------------------------------------------------------
; Integer min / max / abs
; ------------------------------------------------------------
m_min:
        mov     rax, rdi
        cmp     rdi, rsi
        cmovg   rax, rsi
        ret

m_max:
        mov     rax, rdi
        cmp     rdi, rsi
        cmovl   rax, rsi
        ret

m_abs:
        mov     rax, rdi
        sar     rax, 63
        xor     rdi, rax
        sub     rdi, rax
        mov     rax, rdi
        ret

; ------------------------------------------------------------
; Floating point arithmetic (SSE)
; ------------------------------------------------------------
m_fadd:         ; double m_fadd(double a, double b)
        addsd   xmm0, xmm1
        ret

m_fsub:         ; double m_fsub(double a, double b)
        subsd   xmm0, xmm1
        ret

m_fmul:         ; double m_fmul(double a, double b)
        mulsd   xmm0, xmm1
        ret

m_fdiv:         ; double m_fdiv(double a, double b)
        divsd   xmm0, xmm1
        ret

m_fneg:         ; double m_fneg(double a)
        movabs  rax, 0x8000000000000000   ; sign bit mask
        movq    xmm1, rax
        xorpd   xmm0, xmm1                ; flip sign bit -> -a
        ret

m_fabs:         ; double m_fabs(double a)
        movabs  rax, 0x7fffffffffffffff
        movq    xmm1, rax
        andpd   xmm0, xmm1
        ret

; ------------------------------------------------------------
; double f_sqrt(double a)  (float square root, SSE)
; ------------------------------------------------------------
f_sqrt:
        sqrtsd  xmm0, xmm0
        ret

; ------------------------------------------------------------
; double f_sin(double a)  (x87 fsin)
; ------------------------------------------------------------
f_sin:
        sub     rsp, 8
        movq    [rsp], xmm0
        fld qword    [rsp]
        fsin
        fstp qword   [rsp]
        movq    xmm0, [rsp]
        add     rsp, 8
        ret

; ------------------------------------------------------------
; double f_cos(double a)  (x87 fcos)
; ------------------------------------------------------------
f_cos:
        sub     rsp, 8
        movq    [rsp], xmm0
        fld qword    [rsp]
        fcos
        fstp qword   [rsp]
        movq    xmm0, [rsp]
        add     rsp, 8
        ret

; ------------------------------------------------------------
; double f_exp(double x)  (e^x, Taylor series sum x^n/n!, n=0..19)
; ------------------------------------------------------------
f_exp:
        sub     rsp, 16
        movq    [rsp], xmm0             ; x
        movsd   xmm0, qword [rel one_const]   ; result = 1  (term_0)
        movsd   xmm1, qword [rel one_const]   ; term = 1
        movsd   xmm2, qword [rsp]             ; xmm2 = x
        mov     rcx, 1                        ; n = 1 .. 19
.loop:
        mulsd   xmm1, xmm2            ; term *= x
        cvtsi2sd xmm3, rcx            ; n
        divsd   xmm1, xmm3            ; term /= n
        addsd   xmm0, xmm1            ; result += term
        inc     rcx
        cmp     rcx, 20
        jle     .loop
        add     rsp, 16
        ret

; ------------------------------------------------------------
; double f_log(double x)  (natural log, ln x = log2(x) * ln2)
; ------------------------------------------------------------
f_log:
        sub     rsp, 16
        movq    [rsp], xmm0             ; x
        fld     qword [rel one_const]   ; st0 = 1
        fld     qword [rsp]             ; st0 = x, st1 = 1
        fyl2x                           ; st0 = 1 * log2(x) = log2(x)
        fld     qword [rel ln2_const]   ; st0 = ln2, st1 = log2(x)
        fmulp                           ; st0 = log2(x) * ln2 = ln(x)
        fstp    qword [rsp + 8]
        movq    xmm0, [rsp + 8]
        add     rsp, 16
        ret

; ------------------------------------------------------------
; double f_sigmoid(double x)  (1 / (1 + e^-x))
; ------------------------------------------------------------
f_sigmoid:
        sub     rsp, 16
        movq    [rsp], xmm0             ; save x
        movabs  rax, 0x8000000000000000
        movq    xmm1, rax
        xorpd   xmm0, xmm1              ; xmm0 = -x
        call    f_exp                   ; xmm0 = e^-x
        movsd   [rsp + 8], xmm0         ; ex
        ; 1 / (1 + ex)  (pure SSE)
        movsd   xmm1, qword [rel one_const]
        addsd   xmm1, [rsp + 8]         ; 1 + ex
        movsd   xmm0, qword [rel one_const] ; 1
        divsd   xmm0, xmm1              ; 1 / (1 + ex)
        add     rsp, 16
        ret

; ------------------------------------------------------------
; double f_erf(double x)  (error function, A&S 7.1.26 approximation)
; ------------------------------------------------------------
f_erf:
        push    rbx
        sub     rsp, 48
        movq    [rsp], xmm0             ; x
        movabs  rax, 0x7fffffffffffffff
        movq    xmm0, rax
        andpd   xmm0, [rsp]             ; |x| in xmm0
        movsd   [rsp + 40], xmm0        ; save |x|

        ; exx = e^(-x^2)
        movsd   xmm0, [rsp + 40]
        mulsd   xmm0, xmm0              ; x^2
        movabs  rax, 0x8000000000000000
        movq    xmm1, rax
        xorpd   xmm0, xmm1              ; -x^2
        call    f_exp                   ; xmm0 = e^(-x^2)
        movsd   [rsp + 8], xmm0         ; exx

        ; t = 1 / (1 + p*|x|)
        movsd   xmm0, [rsp + 40]        ; |x|
        mulsd   xmm0, qword [rel p_const]
        addsd   xmm0, qword [rel one_const]
        movsd   xmm3, qword [rel one_const]
        divsd   xmm3, xmm0              ; t
        movsd   [rsp + 16], xmm3        ; t

        ; poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        movsd   xmm0, [rsp + 16]        ; t
        mulsd   xmm0, qword [rel a5_const]
        addsd   xmm0, qword [rel a4_const]
        mulsd   xmm0, [rsp + 16]
        addsd   xmm0, qword [rel a3_const]
        mulsd   xmm0, [rsp + 16]
        addsd   xmm0, qword [rel a2_const]
        mulsd   xmm0, [rsp + 16]
        addsd   xmm0, qword [rel a1_const]
        mulsd   xmm0, [rsp + 16]        ; poly

        ; result = 1 - poly * exx
        mulsd   xmm0, [rsp + 8]         ; poly * exx
        movsd   xmm1, qword [rel one_const]
        subsd   xmm1, xmm0              ; erf(|x|)

        ; apply sign of original x
        movabs  rax, 0x8000000000000000
        movq    xmm2, rax
        andpd   xmm2, [rsp]             ; sign bit of x
        xorpd   xmm1, xmm2              ; negate if x < 0

        movsd   [rsp + 32], xmm1
        movq    xmm0, [rsp + 32]
        add     rsp, 48
        pop     rbx
        ret

; ------------------------------------------------------------
; double f_square(double x)  (x^2)
; ------------------------------------------------------------
f_square:
        mulsd   xmm0, xmm0
        ret

; ------------------------------------------------------------
; long i_square(long x)  (x^2, integer)
; ------------------------------------------------------------
i_square:
        mov     rax, rdi
        imul    rax, rax
        ret

; ------------------------------------------------------------
; double f_threshold(double x, double thresh)  (1.0 if x > thresh else 0.0)
;   Args: xmm0 = x, xmm1 = thresh
; ------------------------------------------------------------
f_threshold:
        comisd  xmm0, xmm1              ; compare x vs thresh (sets flags)
        seta    al                      ; al = 1 if x > thresh  (flags still valid)
        movzx   rax, al                 ; rax = 0 or 1
        cvtsi2sd xmm0, rax              ; xmm0 = 1.0 or 0.0
        ret
