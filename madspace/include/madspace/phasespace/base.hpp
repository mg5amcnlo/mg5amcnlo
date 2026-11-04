#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * Base class for invertible phase-space mappings.
 *
 * Cross-section and event-generation integrals have the form
 *
 * @f[
 *   I = \int_\Phi f(x)\,\mathrm{d}x,
 * @f]
 *
 * with @f$f(x)@f$ the fully differential cross section over the phase-space
 * point @f$x@f$ [1]. Monte-Carlo integration draws @f$x@f$ through an invertible
 * map @f$G@f$ from the unit hypercube, @f$r \in [0,1]^d \leftrightarrow x \in
 * \Phi@f$. This induces the normalized sampling density
 *
 * @f[
 *   g(x) = \left| \frac{\partial G^{-1}(x)}{\partial x} \right|,
 *   \qquad \int_\Phi g(x)\,\mathrm{d}x = 1,
 * @f]
 *
 * so that @f$I = \int_U \left. f(x)/g(x) \right|_{x = G(r)}\,\mathrm{d}r@f$. The
 * integration variance shrinks as @f$g@f$ approaches @f$f/I@f$.
 *
 * A `Mapping` records one such @f$G@f$ as a pair of compute-graph builders.
 * `build_forward` maps the **inputs** (the unit-hypercube numbers) to the
 * **outputs** (momenta and invariants). `build_inverse` maps the outputs back
 * to the inputs. Both also return a `det`, the Jacobian @f$1/g@f$ of the
 * transformation. **Conditions** are external quantities the mapping needs but
 * does not transform, such as masses, the collision energy or cut boundaries.
 * They are the same for the forward and inverse direction.
 *
 * Random-number inputs are named after the variable they generate (`r_phi`,
 * `r_theta`, `r_t`, `r_s`, …). Time-like invariants are called `s_i`,
 * space-like ones `t_i`. Entries marked with an index `i` are repeated per
 * particle. `batch` is the leading batch dimension of every tensor. Concrete
 * subclasses list their `Inputs`, `Conditions` and `Outputs` individually.
 *
 * `build_forward` and `build_inverse` each come in two overloads. One takes a
 * `NamedVector<Value>`. The other takes a positional `ValueVec` and attaches
 * the names from `input_types()` / `output_types()` and `condition_types()`.
 * Subclasses may be written in Python by deriving from this class.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class Mapping {
public:
    /// Forward result: the transformed values together with the Jacobian weight.
    using Result = std::tuple<NamedVector<Value>, Value>;

    /**
     * @param name             Identifier used for the generated compute graph.
     * @param input_types       Types of the values consumed by the forward map
     *                          (produced by the inverse map).
     * @param output_types      Types of the values produced by the forward map
     *                          (consumed by the inverse map).
     * @param condition_types   Types of the pass-through conditioning values.
     */
    Mapping(
        const std::string& name,
        const NamedVector<Type>& input_types,
        const NamedVector<Type>& output_types,
        const NamedVector<Type>& condition_types
    ) :
        _name(name),
        _input_types(input_types),
        _output_types(output_types),
        _condition_types(condition_types) {}
    virtual ~Mapping() = default;
    /// Add the forward map to @p fb and return its named outputs. The positional
    /// overload attaches names from `input_types()` and `condition_types()`.
    NamedVector<Value> build_forward(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions = {}
    ) const;
    /// Add the forward map to @p fb and return its named outputs. The positional
    /// overload attaches names from `input_types()` and `condition_types()`.
    NamedVector<Value> build_forward(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions = {}
    ) const {
        return build_forward(
            fb, {_input_types.keys(), inputs}, {_condition_types.keys(), conditions}
        );
    }
    /// Add the inverse map to @p fb and return the recovered inputs. The
    /// positional overload attaches names from `output_types()` and
    /// `condition_types()`.
    NamedVector<Value> build_inverse(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions = {}
    ) const {
        return build_inverse(
            fb, {_output_types.keys(), inputs}, {_condition_types.keys(), conditions}
        );
    }
    /// Add the inverse map to @p fb and return the recovered inputs. The
    /// positional overload attaches names from `output_types()` and
    /// `condition_types()`.
    NamedVector<Value> build_inverse(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions = {}
    ) const;
    /// Build a standalone `Function` for the forward map.
    Function forward_function() const;
    /// Build a standalone `Function` for the inverse map.
    Function inverse_function() const;
    /// Types of the forward inputs / inverse outputs.
    const NamedVector<Type>& input_types() const { return _input_types; }
    /// Types of the forward outputs / inverse inputs.
    const NamedVector<Type>& output_types() const { return _output_types; }
    /// Types of the pass-through conditions.
    const NamedVector<Type>& condition_types() const { return _condition_types; }
    /// Compute-graph identifier passed to the constructor.
    const std::string& name() const { return _name; }
    /// Number of inputs that are discrete (integer) choices rather than
    /// continuous unit-hypercube coordinates.
    virtual std::size_t discrete_dim() const { return 0; }

protected:
    // TODO: make parameters const ref
    virtual Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const = 0;
    virtual Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const = 0;

private:
    std::string _name;
    NamedVector<Type> _input_types;
    NamedVector<Type> _output_types;
    NamedVector<Type> _condition_types;
};

/**
 * Base class for compute-graph functions that are not invertible mappings.
 *
 * A `FunctionGenerator` builds a `Function` from named **Arguments** to named
 * **Returns**. Unlike @ref Mapping it has no inverse and produces no Jacobian
 * weight. It is used for the pieces of the integrand that only need to be
 * evaluated in one direction: cuts, observables, histograms, matrix elements,
 * PDF and scale evaluation, and the trainable networks. Concrete subclasses
 * list their arguments and returns individually. Subclasses may be written in
 * Python by deriving from this class.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class FunctionGenerator {
public:
    /**
     * @param name          Identifier used for the generated compute graph.
     * @param arg_types     Types of the values consumed by the function.
     * @param return_types  Types of the values produced by the function.
     */
    FunctionGenerator(
        const std::string& name,
        const NamedVector<Type>& arg_types,
        const NamedVector<Type>& return_types
    ) :
        _name(name), _arg_types(arg_types), _return_types(return_types) {}
    virtual ~FunctionGenerator() = default;
    /// Add the function to @p fb and return its named results. The positional
    /// overload attaches names from `arg_types()`.
    NamedVector<Value>
    build_function(FunctionBuilder& fb, const NamedVector<Value>& args) const;
    /// Add the function to @p fb and return its named results. The positional
    /// overload attaches names from `arg_types()`.
    NamedVector<Value> build_function(FunctionBuilder& fb, const ValueVec& args) const {
        return build_function(fb, {_arg_types.keys(), args});
    }
    /// Build a standalone `Function`.
    Function function() const;
    /// Types of the function arguments.
    const NamedVector<Type>& arg_types() const { return _arg_types; }
    /// Types of the function returns.
    const NamedVector<Type>& return_types() const { return _return_types; }
    /// Compute-graph identifier passed to the constructor.
    const std::string& name() const { return _name; }

protected:
    virtual NamedVector<Value>
    build_function_impl(FunctionBuilder& fb, const NamedVector<Value>& args) const = 0;

private:
    std::string _name;
    NamedVector<Type> _arg_types;
    NamedVector<Type> _return_types;
};

} // namespace madspace
