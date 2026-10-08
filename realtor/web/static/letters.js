// Written-request templates. The law and the LTB say to raise these in writing; these give a
// plain starting point that the person edits before sending. Nothing is filled in from the
// lease or sent anywhere: brackets mark what to fill in.
const LETTERS = {
  repair: {
    title: "Repair request",
    body: (issue, date) => `${date}

To: [Landlord or property manager]
Re: Repair request for [unit and address]

I'm writing to ask for a repair in my unit.

The problem: ${issue}
It started on: [date]
How it affects the unit: [for example, the door won't lock, so the unit isn't secure]

Please let me know when the repair will be done. If you need to enter the unit, please give written notice, or contact me to arrange a time.

I've attached photos. Please keep this letter on file.

Thank you,
[Your name]
[Phone or email]`,
  },
  entry: {
    title: "Request for notice before entry",
    body: (issue, date) => `${date}

To: [Landlord or property manager]
Re: Entry into [unit and address]

I'm writing about entry into my unit. What happened: ${issue}

Under the Residential Tenancies Act, a landlord must give written notice at least 24 hours before entering, with the reason and a time between 8 a.m. and 8 p.m., except in an emergency or with my consent at the time.

Please give written notice before any future entry.

Thank you,
[Your name]
[Phone or email]`,
  },
  disturbance: {
    title: "Report of a disturbance",
    body: (issue, date) => `${date}

To: [Landlord or property manager]
Re: Ongoing disturbance at [address]

I'm writing to report a disturbance that is affecting my use of my unit.

What's happening: ${issue}
Dates and times: [list them, from your log]
How it affects me: [for example, I can't sleep before 2 a.m.]

Please let me know what steps you'll take to deal with it.

Thank you,
[Your name]
[Phone or email]`,
  },
};
