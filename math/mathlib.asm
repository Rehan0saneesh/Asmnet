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
