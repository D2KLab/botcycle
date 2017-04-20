create table if not exists User (
  id varchar(20) not null,
  name varchar(50),
  surname varchar(50),
  primary key (id)
);

create table if not exists Place (
  id varchar(20) not null,
  name varchar(50) not null,
  description text,
  lat double precision not null,
  lng double precision not null,
  category varchar(50),
  primary key (id)
);

create table if not exists Event (
  userId varchar(20) not null,
  placeId varchar(20) not null,
  time DATETIME not null,
  action varchar(20),
  primary key (userId, placeId, time),
  foreign key (userId) references User(id),
  foreign key (placeId) references Place(id)
);

create table if not exists UserPlace (
  userId varchar(20) not null,
  placeId varchar(20) not null,
  role varchar(50),
  primary key (userId, placeId)
  foreign key (userId) references User(id),
  foreign key (placeId) references Place(id)
)
