from realtor.lease.sources import Source
from realtor.lease.verify import verify

S20 = Source("rta:20", "rta", "Residential Tenancies Act, 2006, s. 20: Landlord's responsibility to repair",
             "(1)  A landlord is responsible for providing and maintaining a residential complex, including the "
             "rental units in it, in a good state of repair and fit for habitation.")
S116 = Source("rta:116", "rta", "Residential Tenancies Act, 2006, s. 116: Notice of rent increase required",
              "(1)  A landlord shall not increase the rent charged to a tenant for a rental unit without first "
              "giving the tenant at least 90 days written notice of the landlord's intention to do so.")


def test_faithful_answer_passes():
    answer = ('**Wear and tear.** RTA s. 20 [1] says: "A landlord is responsible for providing and maintaining a '
              'residential complex, including the rental units in it, in a good state of repair"\n'
              "The landlord needs to give 90 days' written notice before a rent increase [2].")
    r = verify(answer, [S20, S116])
    assert r.ok, r.issues
    assert r.checked_quotes == 1 and r.checked_facts >= 1


def test_altered_quote_is_caught():
    answer = 'RTA s. 20 [1] says: "A landlord is responsible for providing a residential complex in perfect condition"'
    r = verify(answer, [S20, S116])
    assert [i.kind for i in r.issues] == ["quote"]


def test_wrong_number_is_caught():
    r = verify("The landlord must give 60 days' notice before raising the rent [2].", [S20, S116])
    assert any(i.kind == "fact" and "60 day" in i.detail for i in r.issues)


def test_made_up_citation_is_caught():
    r = verify("See [7] for details.", [S20, S116])
    assert [i.kind for i in r.issues] == ["citation"]


def test_pointers_are_checked():
    from realtor.lease.pointers import expand
    ok = verify("**Wear and tear.** RTA s. 20 [1] says: {1.1}", [S20, S116])
    assert ok.ok and ok.checked_quotes == 1
    bad = verify("RTA s. 20 [1] says: {1.4}", [S20, S116])
    assert [i.kind for i in bad.issues] == ["pointer"]
    assert expand("says: {2.1}", [S20, S116]).startswith('says: "(1)  A landlord shall not increase')
