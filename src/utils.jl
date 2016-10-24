
"""
Recursively replace all occurences of a field with dot notation:
```julia
type T
    a
    b
end
@with t::T a = b + 5
# becomes:
t.a = t.b + 5
```
"""
macro with(o, body::Expr)
    _with(o, body)
end

# --------------------------------------------------------------------------

# recurse
function replace_syms(expr::Expr, names, o)
    for i in eachindex(expr.args)
        expr.args[i] = replace_syms(expr.args[i], names, o)
    end
    expr
end

# replace?
function replace_syms(s::Symbol, names, o)
    if s in names
        :($(o).$(s))
    else
        s
    end
end

# identity
replace_syms(x, names, obj) = x

# if we pass in something like: type X; a; end; @with x f(x::X) = 10a
# then we look in the function definition to get the type of x
function _with(o::Symbol, ex::Expr)
    @assert ex.head == :function
    for farg in ex.args[1].args[2:end]
        # farg is an argument expression
        if farg == o
            error("You need to attach a type specification to know how to apply the @with macro, like: type X; a; end; @with x f(x::X) = 10a")
        elseif isa(farg, Expr) && farg.head == :(::) && farg.args[1] == o
            # found the right expression... apply it
            return _with(farg, ex)
        end
    end
    error("Couldn't match the symbol $o to a function argument in $(ex.args[1])")
end

function _with(oex::Expr, ex::Expr)
    # o is the symbol (x) and objtype is the type (X)
    @assert oex.head == :(::)
    @assert length(oex.args) == 2
    o, objtype = oex.args

    # if it's parameterized (x::X{T}) then just get the X-part
    if isa(objtype, Expr)
        @assert objtype.head == :curly
        objtype = objtype.args[1]
    end

    # get the fieldnames of that type, and replace them all
    names = fieldnames(eval(current_module(), objtype))
    esc(replace_syms(ex, names, o))
end
