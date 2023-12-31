"""change column name

Revision ID: 116b546534fa
Revises: edd97c887664
Create Date: 2023-07-17 21:23:04.096373

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '116b546534fa'
down_revision = 'edd97c887664'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('climb', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_username', sa.String(length=64), nullable=True))
        batch_op.drop_constraint('fk_climb_user_id', type_='foreignkey')
        batch_op.create_foreign_key('fk_climb_user_username', 'user', ['user_username'], ['username'])
        batch_op.drop_column('user_id')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('climb', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.INTEGER(), nullable=True))
        batch_op.drop_constraint('fk_climb_user_username', type_='foreignkey')
        batch_op.create_foreign_key('fk_climb_user_id', 'user', ['user_id'], ['id'])
        batch_op.drop_column('user_username')

    # ### end Alembic commands ###
