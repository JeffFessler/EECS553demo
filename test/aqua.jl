using Name: Name
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_ambiguities(Name) # if isolation needed
    Aqua.test_all(Name; ambiguities = false)
end
